"""Register the FieldOps CRM operations as declarative tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from capstone.core.crm import CRMRepository
from capstone.tools.recipes import make_recipe_impl
from capstone.tools.registry import ToolRegistry, ToolSpec

if TYPE_CHECKING:
    from capstone.ml.serving import ModelService

CATEGORY_ENUM = ["plumbing", "electrical", "hvac", "gas", "fire_safety", "general_maintenance"]
URGENCY_ENUM = ["emergency", "high", "normal", "low"]


def build_registry(crm: CRMRepository, models: ModelService | None = None) -> ToolRegistry:
    reg = ToolRegistry()
    if models is not None:
        from capstone.tools.ml_tools import register_ml_tools

        register_ml_tools(reg, crm, models)
    reg.register(ToolSpec(
        name="lookup_customer",
        description="Find the customer record by phone number or email. Returns sites and assets.",
        kind="python",
        input_schema={"type": "object", "additionalProperties": False,
                      "properties": {"phone": {"type": "string"}, "email": {"type": "string"}}}),
        lambda t, a: crm.find_customer(t, **a))
    reg.register(ToolSpec(
        name="find_available_slots",
        description="List the next available engineer slots for a trade category and urgency.",
        kind="python",
        input_schema={"type": "object", "required": ["category", "urgency"],
                      "properties": {"category": {"type": "string", "enum": CATEGORY_ENUM},
                                     "urgency": {"type": "string", "enum": URGENCY_ENUM}}}),
        lambda t, a: crm.find_available_slots(t, a["category"], a["urgency"]))
    reg.register(ToolSpec(
        name="get_open_jobs",
        description="List the customer's scheduled (not yet completed) jobs.",
        kind="python",
        input_schema={"type": "object", "required": ["customer_id"],
                      "properties": {"customer_id": {"type": "string"}}}),
        lambda t, a: crm.get_open_jobs(t, a["customer_id"]))
    reg.register(ToolSpec(
        name="job_history",
        description="Most recent jobs for a customer, newest first, for context on repeat faults.",
        kind="python",
        input_schema={"type": "object", "required": ["customer_id"],
                      "properties": {"customer_id": {"type": "string"}}}),
        lambda t, a: crm.job_history(t, a["customer_id"]))
    reg.register(ToolSpec(
        name="create_job",
        description="Create and schedule a job in a specific slot. Side effect: books an engineer.",
        kind="python", side_effects=True, requires_approval=True,
        input_schema={"type": "object",
                      "required": ["category", "urgency", "description", "slot"],
                      "properties": {"customer_id": {"type": "string"},
                                     "site_id": {"type": "string"},
                                     "address": {"type": "string"},
                                     "category": {"type": "string", "enum": CATEGORY_ENUM},
                                     "urgency": {"type": "string", "enum": URGENCY_ENUM},
                                     "description": {"type": "string"},
                                     "slot": {"type": "object"}}}),
        lambda t, a: crm.create_job(t, customer_id=a.get("customer_id"),
                                    site_id=a.get("site_id"), address=a.get("address"),
                                    category=a["category"], urgency=a["urgency"],
                                    description=a["description"], slot=a["slot"]))
    reschedule = ToolSpec(
        name="reschedule_options",
        description="Recipe: find a customer's open job and propose alternative slots for it.",
        kind="recipe",
        input_schema={"type": "object", "required": ["customer_id"],
                      "properties": {"customer_id": {"type": "string"}}},
        config={"steps": [
            {"id": "jobs", "tool": "get_open_jobs",
             "args": {"customer_id": "{{input.customer_id}}"}},
            {"id": "slots", "tool": "find_available_slots",
             "args": {"category": "{{steps.jobs.0.category}}", "urgency": "normal"}},
        ], "output": {"job": "{{steps.jobs.0}}", "options": "{{steps.slots}}"}})
    reg.register(reschedule, make_recipe_impl(reschedule, reg))
    return reg
