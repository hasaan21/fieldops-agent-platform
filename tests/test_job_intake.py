from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from capstone.agents.job_intake.graph import IntakeContext, IntakeDeps, build_graph
from capstone.agents.memory import load_memories


def make_app(crm):
    return build_graph(IntakeDeps(crm=crm)).compile(checkpointer=InMemorySaver(),
                                                    store=InMemoryStore())


def run(app, text, ctx, thread="th-1"):
    return app.invoke({"messages": [HumanMessage(text)]},
                      {"configurable": {"thread_id": thread}}, context=ctx)


def test_booking_interrupts_for_human_review_then_books(crm, known_customer):
    app = make_app(crm)
    ctx = IntakeContext(tenant_id="t1", channel="sms", sender=known_customer["phone"])
    out = run(app, "The Thermacore X200 boiler is showing error code F22, can you send someone?",
              ctx)
    assert "__interrupt__" in out
    payload = out["__interrupt__"][0].value
    assert payload["type"] == "booking_approval"
    assert payload["proposal"]["customer_id"] == known_customer["customer_id"]

    out = app.invoke(Command(resume={"approved": True, "reviewer": "ops@t1"}),
                     {"configurable": {"thread_id": "th-1"}}, context=ctx)
    assert out["outcome"] == "booked"
    assert out["trajectory"] == ["lookup_customer", "find_available_slots", "create_job"]
    assert out["job"]["tenant_id"] == "t1"
    mems = load_memories(app.store, "t1", known_customer["customer_id"])
    assert any(out["job"]["job_id"] in m for m in mems)


def test_reviewer_rejection_does_not_book(crm, known_customer):
    app = make_app(crm)
    ctx = IntakeContext(tenant_id="t1", sender=known_customer["phone"])
    run(app, "No hot water, the Aquaflow WH-80 shows W1", ctx, "th-2")
    out = app.invoke(Command(resume={"approved": False}), {"configurable": {"thread_id": "th-2"}},
                     context=ctx)
    assert out["outcome"] == "rejected_by_reviewer"
    assert "create_job" not in out["trajectory"]


def test_gas_emergency_escalates_and_never_books(crm, known_customer):
    app = make_app(crm)
    ctx = IntakeContext(tenant_id="t1", sender=known_customer["phone"], review_mode="auto_approve")
    out = run(app, "I can smell gas near the boiler in the kitchen!", ctx, "th-3")
    assert out["outcome"] == "escalated"
    assert "create_job" not in out["trajectory"]
    assert "turn off the supply" in out["reply"]


def test_unknown_sender_without_address_is_asked_to_clarify(crm):
    app = make_app(crm)
    ctx = IntakeContext(tenant_id="t1", sender="+447000000000", review_mode="auto_approve")
    out = run(app, "Our boiler is showing F28 and won't light", ctx, "th-4")
    assert out["outcome"] == "clarify"
    assert "address" in out["reply"]


def test_card_numbers_are_redacted_before_classification_and_in_reply(crm, known_customer):
    app = make_app(crm)
    ctx = IntakeContext(tenant_id="t1", sender=known_customer["phone"], review_mode="auto_approve")
    out = run(app, "Tap dripping in the toilet. My card is 4111 1111 1111 1111 exp 09/28", ctx,
              "th-5")
    assert all("4111" not in str(m.content) for m in out["messages"])
    assert "redacted:card_number" in out["flags"]


def test_question_uses_tenant_knowledge_base(crm):
    app = make_app(crm)
    out = run(app, "How much is your call-out fee?", IntakeContext(tenant_id="t2"), "th-6")
    assert out["outcome"] == "answered"
    assert "80" in out["reply"]  # t2's fee; t1 is 65, t3 is 95
    assert "search_knowledge_base" in out["trajectory"]


def test_out_of_scope_trade_is_declined(crm):
    app = make_app(crm)
    ctx = IntakeContext(tenant_id="t2", review_mode="auto_approve")  # t2: electrical + fire only
    out = run(app, "Water is pouring through the ceiling from the water heater cupboard!", ctx,
              "th-7")
    assert out["outcome"] == "declined"
