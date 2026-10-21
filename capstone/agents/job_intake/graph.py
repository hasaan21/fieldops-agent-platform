"""Job Intake agent: an explicit LangGraph state machine wrapped around LLM steps.

    guard_input -> load_context -> classify --route--> decline | answer | reschedule |
                                                      escalate | clarify | plan_booking
    plan_booking -> human_review (interrupt) -> book_job -> respond -> END

Design choices worth defending in an interview:
- Explicit graph for the business-critical path (booking), with a ReAct loop only for open-ended
  Q&A (`answer`). The LLM classifies and phrases; deterministic code decides what happens.
- Tenant, channel and sender are runtime *context* (immutable, server-set). The LLM can't change
  them.
- Booking needs approval: an `interrupt()` unless the tenant's policy auto-approves.
- Every tool call is appended to `trajectory`, so evals can check the path taken, not just the
  final answer.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, trim_messages
from langgraph.config import get_config
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from langgraph.types import interrupt

from capstone.agents.job_intake.answerer import Answerer, RetrievalAnswerer
from capstone.agents.job_intake.classifier import Classifier, RuleBasedClassifier
from capstone.agents.job_intake.responder import Responder, TemplateResponder
from capstone.agents.job_intake.schemas import BookingProposal, Classification, ReviewDecision
from capstone.agents.memory import load_memories, save_memory
from capstone.core.crm import CRMRepository
from capstone.data.synthetic.catalog import ASSET_TYPES
from capstone.guardrails import injection, output_policy, pii, safety

if TYPE_CHECKING:
    from capstone.ml.serving import ModelService

SAFETY_CATEGORIES = {"gas", "electrical"}
URGENCY_LEVELS = ["low", "normal", "high", "emergency"]
ML_UPGRADE_CONFIDENCE = 0.7


@dataclass(frozen=True)
class IntakeContext:
    tenant_id: str
    channel: str = "sms"
    sender: str = ""
    review_mode: Literal["policy", "auto_approve"] = "policy"


class IntakeState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    customer: dict[str, Any] | None
    memories: list[str]
    classification: dict[str, Any]
    flags: Annotated[list[str], operator.add]
    proposal: dict[str, Any] | None
    review: dict[str, Any] | None
    job: dict[str, Any] | None
    answer: dict[str, Any] | None
    outcome: str
    reply: str
    trajectory: Annotated[list[str], operator.add]


@dataclass
class IntakeDeps:
    crm: CRMRepository
    classifier: Classifier = field(default_factory=RuleBasedClassifier)
    responder: Responder = field(default_factory=TemplateResponder)
    answerer: Answerer | None = None
    models: ModelService | None = None  # ML tools; None = agent runs without them

    def get_answerer(self) -> Answerer:
        if self.answerer is None:
            self.answerer = RetrievalAnswerer.from_synthetic()
        return self.answerer


def _last_human(state: IntakeState) -> HumanMessage:
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            return m
    raise ValueError("no human message in state")


def build_graph(deps: IntakeDeps) -> StateGraph[IntakeState, IntakeContext, IntakeState,
                                                IntakeState]:
    crm = deps.crm

    def guard_input(state: IntakeState, runtime: Runtime[IntakeContext]) -> dict[str, Any]:
        msg = _last_human(state)
        red = pii.redact(str(msg.content))
        scan = injection.scan(red.text)
        flags = [f"redacted:{r}" for r in red.redactions]
        if scan.flagged:
            flags.append("injection_suspected")
        update: dict[str, Any] = {"flags": flags}
        if red.redactions:
            update["messages"] = [HumanMessage(content=red.text, id=msg.id)]
        return update

    def load_context(state: IntakeState, runtime: Runtime[IntakeContext]) -> dict[str, Any]:
        ctx = runtime.context
        by = {"email": ctx.sender} if "@" in ctx.sender else {"phone": ctx.sender}
        customer = crm.find_customer(ctx.tenant_id, **by) if ctx.sender else None
        memories: list[str] = []
        if customer:
            if customer.get("preferences"):
                memories.append(customer["preferences"])
            if runtime.store is not None:
                memories += load_memories(runtime.store, ctx.tenant_id, customer["customer_id"])
        return {"customer": customer, "memories": memories, "trajectory": ["lookup_customer"]}

    def classify(state: IntakeState, runtime: Runtime[IntakeContext]) -> dict[str, Any]:
        ctx = runtime.context
        tenant = crm.tenant(ctx.tenant_id)
        msgs = state["messages"]
        history = trim_messages(msgs[:-1], max_tokens=8, token_counter=len, strategy="last",
                                start_on="human", include_system=False)
        c = deps.classifier.classify(str(_last_human(state).content), tenant=tenant,
                                     customer=state.get("customer"), channel=ctx.channel,
                                     history=history)
        flags = ["injection_suspected"] if c.injection_attempt else []
        trajectory: list[str] = []
        if deps.models is not None and c.intent == "book_job":
            # Second opinion from the cheap text model. Asymmetric policy: it may only
            # *upgrade* to emergency (under-triage is the costly error); other disagreements
            # are flagged for the reviewer, not acted on.
            opinion = deps.models.urgency(str(_last_human(state).content))
            if opinion:
                trajectory.append("ml_urgency")
                ml_u = opinion["urgency"]
                if ml_u == "emergency" and c.urgency != "emergency" and \
                        opinion.get("confidence", 0) >= ML_UPGRADE_CONFIDENCE:
                    c = c.model_copy(update={"urgency": "emergency"})
                    flags.append("ml_urgency_upgrade")
                elif abs(URGENCY_LEVELS.index(ml_u) - URGENCY_LEVELS.index(c.urgency)) >= 2:
                    flags.append(f"ml_urgency_disagree:{ml_u}")
        return {"classification": c.model_dump(), "flags": flags, "trajectory": trajectory}

    def route(state: IntakeState, runtime: Runtime[IntakeContext]) -> str:
        tenant = crm.tenant(runtime.context.tenant_id)
        c = Classification(**state["classification"])
        if c.intent == "out_of_scope" or (
                c.intent == "book_job" and c.category != "unknown"
                and c.category not in tenant.trades):
            return "decline"
        if c.intent == "question":
            return "answer"
        if c.intent == "reschedule":
            return "reschedule"
        if c.safety_risk or (c.urgency == "emergency" and c.category in SAFETY_CATEGORIES) or \
                safety.life_safety_risk(str(_last_human(state).content)):
            return "escalate"
        customer = state.get("customer") or {}
        has_location = bool(customer.get("sites")) or bool(c.address or c.postcode)
        if c.category == "unknown" or not has_location:
            return "clarify"
        return "plan_booking"

    def decline(state: IntakeState) -> dict[str, Any]:
        return {"outcome": "declined"}

    def clarify(state: IntakeState) -> dict[str, Any]:
        return {"outcome": "clarify"}

    def escalate(state: IntakeState) -> dict[str, Any]:
        return {"outcome": "escalated", "trajectory": ["escalate_to_human"]}

    def reschedule(state: IntakeState, runtime: Runtime[IntakeContext]) -> dict[str, Any]:
        customer = state.get("customer")
        jobs = crm.get_open_jobs(runtime.context.tenant_id, customer["customer_id"]) \
            if customer else []
        return {"outcome": "reschedule_handoff", "trajectory": ["get_open_jobs"],
                "flags": [f"open_jobs:{len(jobs)}"]}

    def answer(state: IntakeState, runtime: Runtime[IntakeContext]) -> dict[str, Any]:
        result = deps.get_answerer().answer(str(_last_human(state).content),
                                            tenant_id=runtime.context.tenant_id)
        return {"outcome": "answered", "answer": result.model_dump(),
                "trajectory": result.tools_called}

    def plan_booking(state: IntakeState, runtime: Runtime[IntakeContext]) -> dict[str, Any]:
        ctx = runtime.context
        c = Classification(**state["classification"])
        slots = crm.find_available_slots(ctx.tenant_id, c.category, c.urgency)
        if not slots:
            return {"outcome": "escalated", "trajectory": ["find_available_slots"],
                    "flags": ["no_slots"]}
        customer = state.get("customer")
        site = None
        if customer and customer.get("sites"):
            site = next((s for s in customer["sites"] if c.postcode and
                         s["postcode"].replace(" ", "") == c.postcode.replace(" ", "")),
                        customer["sites"][0])
        description = str(_last_human(state).content)[:500]
        asset = None
        if customer:
            asset = next((a for a in customer.get("assets", [])
                          if ASSET_TYPES.get(a["asset_type"], ("",))[0] == c.category
                          and (site is None or a["site_id"] == site["site_id"])), None)
        proposal = BookingProposal(
            customer_id=customer["customer_id"] if customer else None,
            site_id=site["site_id"] if site else None,
            address=site["address"] if site else " ".join(filter(None, [c.address, c.postcode])),
            category=c.category, urgency=c.urgency, description=description,
            slot=slots[0], alternatives=slots[1:],
            asset_id=asset["asset_id"] if asset else None,
        )
        trajectory = ["find_available_slots"]
        flags: list[str] = []
        if deps.models is not None:
            X = deps.models.duration_features(
                tenant_id=ctx.tenant_id, category=c.category, urgency=c.urgency,
                description=description, asset=asset,
                when=datetime.fromisoformat(slots[0]["start"]))
            proposal.estimated_hours = deps.models.estimate_duration(X)
            trajectory.append("estimate_job_duration")
            if asset and (snap := deps.models.latest_snapshot(asset["asset_id"])) is not None:
                proposal.asset_risk = deps.models.failure_risk(snap)
                trajectory.append("asset_failure_risk")
                if proposal.asset_risk["band"] == "high":
                    flags.append("maintenance_suggested")
        return {"proposal": proposal.model_dump(), "trajectory": trajectory, "flags": flags}

    def human_review(state: IntakeState, runtime: Runtime[IntakeContext]) -> dict[str, Any]:
        ctx = runtime.context
        tenant = crm.tenant(ctx.tenant_id)
        if ctx.review_mode == "auto_approve" or tenant.auto_approve_bookings:
            return {"review": {"approved": True, "reviewer": "policy"}}
        raw = interrupt({
            "type": "booking_approval",
            "tenant_id": ctx.tenant_id,
            "proposal": state["proposal"],
            "flags": state.get("flags", []),
            "message": str(_last_human(state).content),
        })
        decision = ReviewDecision.model_validate(raw)
        update: dict[str, Any] = {"review": decision.model_dump()}
        if not decision.approved:
            update["outcome"] = "rejected_by_reviewer"
        elif decision.slot:
            update["proposal"] = {**(state["proposal"] or {}), "slot": decision.slot}
        return update

    def after_review(state: IntakeState) -> str:
        return "respond" if state.get("outcome") == "rejected_by_reviewer" else "book_job"

    def book_job(state: IntakeState, runtime: Runtime[IntakeContext]) -> dict[str, Any]:
        ctx = runtime.context
        p = BookingProposal(**state["proposal"])  # type: ignore[arg-type]
        thread_id = get_config().get("configurable", {}).get("thread_id", "no-thread")
        # Slots can be taken between proposal and approval (concurrent customers, slow
        # reviewers): fall back to alternatives, then re-query, before giving up.
        candidates = [p.slot, *p.alternatives]
        candidates += crm.find_available_slots(ctx.tenant_id, p.category, p.urgency, limit=5)
        job = None
        for slot in candidates:
            try:
                job = crm.create_job(ctx.tenant_id, customer_id=p.customer_id, site_id=p.site_id,
                                     category=p.category, urgency=p.urgency,
                                     description=p.description, slot=slot, address=p.address,
                                     idempotency_key=f"{thread_id}:{_last_human(state).id}")
                p.slot = slot
                break
            except ValueError:
                continue
        if job is None:
            return {"outcome": "escalated", "flags": ["slot_conflict"],
                    "trajectory": ["create_job_failed"]}
        if p.customer_id and runtime.store is not None:
            save_memory(runtime.store, ctx.tenant_id, p.customer_id, job["job_id"],
                        f"Booked {p.category} job {job['job_id']} for {p.slot['start']}")
        return {"job": job, "proposal": p.model_dump(), "outcome": "booked",
                "trajectory": ["create_job"]}

    def respond(state: IntakeState, runtime: Runtime[IntakeContext]) -> dict[str, Any]:
        ctx = runtime.context
        tenant = crm.tenant(ctx.tenant_id)
        outcome = state.get("outcome", "clarify")
        c = state.get("classification", {})
        facts: dict[str, Any] = {
            "company": tenant.name,
            "emergency_line": tenant.facts.get("emergency_line"),
            "hours": tenant.facts.get("hours"),
        }
        if customer := state.get("customer"):
            facts["customer_first_name"] = customer["name"].split()[0]
        if state.get("memories"):
            facts["customer_notes"] = state["memories"]
        if outcome == "booked" and (job := state.get("job")) and \
                (proposal := state.get("proposal")):
            facts |= {"job_id": job["job_id"], "slot": proposal["slot"],
                      "category": c.get("category")}
        if outcome == "clarify":
            missing = list(c.get("missing_info") or [])
            if not state.get("customer") and not (c.get("address") or c.get("postcode")):
                missing = sorted(set(missing) | {"address"})
            if c.get("category") == "unknown":
                missing = sorted(set(missing) | {"asset"})
            facts["missing_info"] = [{"address": "site address and postcode",
                                      "asset": "equipment that has the problem"}.get(m, m)
                                     for m in missing]
        if outcome == "answered" and (answer := state.get("answer")):
            facts |= {"answer": answer["text"], "sources": answer["sources"]}
        message = str(_last_human(state).content)
        reply = deps.responder.respond(tenant=tenant, channel=ctx.channel, outcome=outcome,
                                       message=message, facts=facts)
        reply = pii.redact(reply).text
        policy = output_policy.check_reply(reply, facts)
        flags = [f"policy:{v}" for v in policy.violations]
        if not policy.ok:
            reply = TemplateResponder().respond(tenant=tenant, channel=ctx.channel,
                                                outcome=outcome, message=message, facts=facts)
        return {"reply": reply, "outcome": outcome, "flags": flags,
                "messages": [AIMessage(content=reply)]}

    g = StateGraph(IntakeState, context_schema=IntakeContext)
    nodes: list[tuple[str, Any]] = [("guard_input", guard_input), ("load_context", load_context),
                     ("classify", classify), ("decline", decline), ("clarify", clarify),
                     ("escalate", escalate), ("reschedule", reschedule), ("answer", answer),
                     ("plan_booking", plan_booking), ("human_review", human_review),
                     ("book_job", book_job), ("respond", respond)]
    for name, fn in nodes:
        g.add_node(name, fn)
    g.add_edge(START, "guard_input")
    g.add_edge("guard_input", "load_context")
    g.add_edge("load_context", "classify")
    g.add_conditional_edges("classify", route, ["decline", "answer", "reschedule", "escalate",
                                                "clarify", "plan_booking"])
    for terminal in ["decline", "answer", "reschedule", "escalate", "clarify"]:
        g.add_edge(terminal, "respond")
    g.add_conditional_edges("plan_booking",
                            lambda s: "respond" if s.get("outcome") else "human_review",
                            ["respond", "human_review"])
    g.add_conditional_edges("human_review", after_review, ["respond", "book_job"])
    g.add_edge("book_job", "respond")
    g.add_edge("respond", END)
    return g


def default_deps(crm: CRMRepository | None = None, *, prompt_version: str | None = None,
                 offline: bool | None = None) -> IntakeDeps:
    from capstone.core.config import get_settings

    crm = crm or CRMRepository()
    s = get_settings()
    use_llm = s.has_llm if offline is None else not offline
    if not use_llm:
        return IntakeDeps(crm=crm)
    from capstone.agents.job_intake.answerer import ReActAnswerer
    from capstone.agents.job_intake.classifier import LLMClassifier
    from capstone.agents.job_intake.responder import LLMResponder
    from capstone.core.llm import get_chat_model

    model = get_chat_model("agent")
    return IntakeDeps(crm=crm, classifier=LLMClassifier(model, prompt_version),
                      responder=LLMResponder(model), answerer=ReActAnswerer.from_synthetic(model))
