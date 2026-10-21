"""PromptFoo Python provider: runs the full Job Intake graph for each test case.

PromptFoo calls `call_api(prompt, options, context)`; `context["vars"]` holds the test vars.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

_target = None


def _get_target():
    global _target
    if _target is None:
        from capstone.evals.target import IntakeTarget

        offline = os.environ.get("FIELDOPS_OFFLINE", "false").lower() == "true"
        _target = IntakeTarget(prompt_version=os.environ.get("FIELDOPS_PROMPT_VERSION"),
                               offline=offline or None)
    return _target


def call_api(prompt: str, options: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    v = context.get("vars", {})
    out = _get_target()({"tenant_id": v.get("tenant_id", "t1"),
                         "channel": v.get("channel", "sms"),
                         "sender": v.get("sender", ""), "message": v.get("message", prompt)})
    return {"output": out.get("reply") or "",
            "metadata": {"outcome": out.get("outcome"), "trajectory": out.get("trajectory"),
                         "classification": json.dumps(out.get("classification"))}}
