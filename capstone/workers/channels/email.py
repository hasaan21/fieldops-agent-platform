"""Email via Nylas v3 (webhook in, send API out), plus reply-cleaning for LLM input.

Emails are the noisiest channel: quoted history, signatures, disclaimers, HTML. Feeding all of
that to the classifier hurts accuracy and opens injection vectors, so strip it first.
"""

from __future__ import annotations

import hashlib
import hmac
import html
import re
from typing import Any

import httpx

from capstone.workers.channels.base import InboundMessage, NumberDirectory

QUOTE_MARKERS = [
    re.compile(r"^On .+wrote:\s*$", re.M),
    re.compile(r"^-{2,}\s*Original Message\s*-{2,}", re.M | re.I),
    re.compile(r"^From: .+$", re.M),
    re.compile(r"^Sent from my (iPhone|Android|mobile)", re.M | re.I),
    re.compile(r"^--\s*$", re.M),  # signature delimiter
]


def html_to_text(body: str) -> str:
    body = re.sub(r"(?is)<(script|style).*?</\1>", "", body)
    body = re.sub(r"(?i)<br\s*/?>|</p>|</div>", "\n", body)
    return html.unescape(re.sub(r"<[^>]+>", "", body))


def clean_reply(body: str) -> str:
    text = html_to_text(body) if "<" in body and ">" in body else body
    cut = min((m.start() for rx in QUOTE_MARKERS if (m := rx.search(text))), default=len(text))
    text = text[:cut]
    lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith(">")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def valid_nylas_signature(secret: str, raw_body: bytes, signature: str) -> bool:
    expected = hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature or "")


def parse_nylas_message(msg: dict[str, Any], directory: NumberDirectory) -> InboundMessage:
    to = msg["to"][0]["email"]
    return InboundMessage(
        provider="nylas", channel="email", tenant_id=directory.tenant_for(to),
        sender=msg["from"][0]["email"], recipient=to,
        text=(f"Subject: {msg.get('subject', '')}\n\n" + clean_reply(msg.get("body", ""))).strip(),
        message_id=msg["id"], thread_ref=msg.get("thread_id"), raw=msg)


class NylasSender:
    def __init__(self, api_key: str, grant_id: str, client: httpx.Client | None = None) -> None:
        self.key, self.grant = api_key, grant_id
        self.http = client or httpx.Client(base_url="https://api.us.nylas.com", timeout=15)

    def send(self, *, to: str, text: str, from_: str, thread_ref: str | None = None) -> str:
        payload: dict[str, Any] = {"to": [{"email": to}], "body": text.replace("\n", "<br>"),
                                   "subject": "Re: your request"}
        if thread_ref:
            payload["reply_to_message_id"] = thread_ref
        r = self.http.post(f"/v3/grants/{self.grant}/messages/send", json=payload,
                           headers={"Authorization": f"Bearer {self.key}"})
        r.raise_for_status()
        return str(r.json()["data"]["id"])
