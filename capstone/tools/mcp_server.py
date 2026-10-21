"""MCP server exposing the FieldOps CRM, so any MCP client (Claude, Cursor, other agents) can use
the same tools our LangGraph agents use.

    FIELDOPS_MCP_TENANT=t1 uv run python -m capstone.tools.mcp_server                  # stdio
    FIELDOPS_MCP_TENANT=t1 uv run python -m capstone.tools.mcp_server --http  # streamable HTTP

Tenant binding: one tenant per stdio process (from env). Over HTTP in production you would pass
a `token_verifier` that maps the bearer token to a tenant, and read it per request from the
authenticated principal. Never accept a tenant id as a tool argument.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

from mcp.server.mcpserver import MCPServer

from capstone.core.crm import CRMRepository
from capstone.tools.crm_tools import build_registry
from capstone.tools.registry import ToolRegistry


def build_server(tenant_id: str, crm: CRMRepository | None = None,
                 allow_side_effects: bool = False) -> tuple[MCPServer, ToolRegistry]:
    crm = crm or CRMRepository()
    reg = build_registry(crm)
    names = [n for n, s in reg.specs.items() if allow_side_effects or not s.side_effects]
    reg.enable(tenant_id, *names)
    server = MCPServer(name=f"fieldops-crm-{tenant_id}",
                       instructions="FieldOps CRM tools for one field-service company. "
                                    "Customer data is confidential.")

    @server.tool(description=reg.specs["lookup_customer"].description)
    def lookup_customer(phone: str | None = None, email: str | None = None) -> dict[str, Any]:
        args = {k: v for k, v in {"phone": phone, "email": email}.items() if v}
        return reg.call(tenant_id, "lookup_customer", args) or {"found": False}

    @server.tool(description=reg.specs["find_available_slots"].description)
    def find_available_slots(category: str, urgency: str = "normal") -> list[dict[str, Any]]:
        return reg.call(tenant_id, "find_available_slots",
                        {"category": category, "urgency": urgency})

    @server.tool(description=reg.specs["job_history"].description)
    def job_history(customer_id: str) -> list[dict[str, Any]]:
        return reg.call(tenant_id, "job_history", {"customer_id": customer_id})

    @server.tool(description=reg.specs["reschedule_options"].description)
    def reschedule_options(customer_id: str) -> dict[str, Any]:
        return reg.call(tenant_id, "reschedule_options", {"customer_id": customer_id})

    if allow_side_effects:
        @server.tool(description=reg.specs["create_job"].description)
        def create_job(category: str, urgency: str, description: str, slot: dict[str, str],
                       customer_id: str | None = None, site_id: str | None = None,
                       address: str | None = None) -> dict[str, Any]:
            args = {"category": category, "urgency": urgency, "description": description,
                    "slot": slot, "customer_id": customer_id, "site_id": site_id,
                    "address": address}
            return reg.call(tenant_id, "create_job",
                            {k: v for k, v in args.items() if v is not None})

    return server, reg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--http", action="store_true")
    args = ap.parse_args()
    server, _ = build_server(os.environ.get("FIELDOPS_MCP_TENANT", "t1"))
    server.run("streamable-http" if args.http else "stdio")


if __name__ == "__main__":
    main()
