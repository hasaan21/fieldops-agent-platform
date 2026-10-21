"""HTTP tools: call third-party or internal REST APIs from a declarative spec.

    config:
      method: GET
      url: https://api.example.com/v1/customers/{customer_id}/invoices
      allowed_hosts: [api.example.com]
      auth: {type: bearer, secret_ref: XERO_TOKEN}     # resolved server-side, never in the prompt
      query: [status]                                   # args sent as query params
      max_response_chars: 4000

Hardening: host allowlist (SSRF), retry only idempotent methods (or with an idempotency key),
exponential backoff on 429/5xx/timeouts, response truncation, no secrets in errors.
"""

from __future__ import annotations

import json
import os
import string
import uuid
from collections.abc import Callable
from typing import Any
from urllib.parse import quote, urlparse

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from capstone.tools.registry import Impl, ToolSpec

IDEMPOTENT = {"GET", "HEAD", "PUT", "DELETE", "OPTIONS"}


class ToolHTTPError(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    return isinstance(exc, ToolHTTPError) and (exc.status == 429 or exc.status >= 500)


def default_secret_resolver(tenant_id: str, ref: str) -> str:
    # Production: a tenant-scoped vault lookup (Azure Key Vault / AWS Secrets Manager).
    return os.environ.get(f"{ref}__{tenant_id.upper()}") or os.environ.get(ref, "")


def make_http_impl(spec: ToolSpec, client: httpx.Client | None = None,
                   secrets: Callable[[str, str], str] = default_secret_resolver) -> Impl:
    cfg = spec.config
    method = cfg.get("method", "GET").upper()
    allowed = set(cfg.get("allowed_hosts", []))
    path_fields = {f for _, f, _, _ in string.Formatter().parse(cfg["url"]) if f}
    http = client or httpx.Client(timeout=spec.timeout_s)

    def impl(tenant_id: str, args: dict[str, Any]) -> Any:
        url = cfg["url"].format(**{k: quote(str(args[k]), safe="") for k in path_fields})
        host = urlparse(url).hostname or ""
        if host not in allowed:
            raise PermissionError(f"host {host!r} not in allowlist")
        headers = {"user-agent": "fieldops-agent/1.0", "x-tenant-id": tenant_id}
        auth = cfg.get("auth")
        if auth:
            token = secrets(tenant_id, auth["secret_ref"])
            if auth.get("type", "bearer") == "bearer":
                headers["authorization"] = f"Bearer {token}"
            else:
                headers[auth.get("header", "x-api-key")] = token
        if method not in IDEMPOTENT:
            headers["idempotency-key"] = args.get("_idempotency_key") or uuid.uuid4().hex
        query = {k: args[k] for k in cfg.get("query", []) if k in args}
        body = {k: v for k, v in args.items()
                if k not in path_fields and k not in query and not k.startswith("_")}

        @retry(retry=retry_if_exception(_retryable), stop=stop_after_attempt(
            cfg.get("max_attempts", 3)), wait=wait_exponential_jitter(multiplier=0.5, max=8),
            reraise=True)
        def send() -> httpx.Response:
            resp = http.request(method, url, params=query or None, headers=headers,
                                json=body if method not in {"GET", "HEAD"} and body else None)
            if resp.status_code >= 400:
                raise ToolHTTPError(resp.status_code, f"{spec.name}: HTTP {resp.status_code}")
            return resp

        resp = send()
        limit = cfg.get("max_response_chars", 4000)
        try:
            text = json.dumps(resp.json())
        except ValueError:
            text = resp.text
        return text[:limit] + ("...[truncated]" if len(text) > limit else "")

    return impl
