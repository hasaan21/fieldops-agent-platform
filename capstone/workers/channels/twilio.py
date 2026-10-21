"""Twilio SMS / WhatsApp / Voice.

Signature validation follows Twilio's spec: base64(HMAC-SHA1(auth_token, full_url + concatenated
sorted POST params)). It's implemented here (not via the SDK) so you can explain it.
https://www.twilio.com/docs/usage/webhooks/webhooks-security
"""

from __future__ import annotations

import base64
import hashlib
import hmac
from collections.abc import Mapping
from typing import Literal
from xml.sax.saxutils import escape

import httpx

from capstone.workers.channels.base import InboundMessage, NumberDirectory


def compute_signature(auth_token: str, url: str, params: Mapping[str, str]) -> str:
    payload = url + "".join(f"{k}{params[k]}" for k in sorted(params))
    digest = hmac.new(auth_token.encode(), payload.encode(), hashlib.sha1).digest()
    return base64.b64encode(digest).decode()


def valid_signature(auth_token: str, url: str, params: Mapping[str, str], signature: str) -> bool:
    return hmac.compare_digest(compute_signature(auth_token, url, params), signature or "")


def parse_message(form: Mapping[str, str], directory: NumberDirectory) -> InboundMessage:
    to, from_ = form["To"], form["From"]
    channel: Literal["whatsapp", "sms"] = "whatsapp" if from_.startswith("whatsapp:") else "sms"
    return InboundMessage(
        provider="twilio", channel=channel, tenant_id=directory.tenant_for(to),
        sender=from_.removeprefix("whatsapp:"), recipient=to, text=form.get("Body", ""),
        message_id=form["MessageSid"], raw=dict(form))


def parse_voice_speech(form: Mapping[str, str], directory: NumberDirectory) -> InboundMessage:
    return InboundMessage(
        provider="twilio-voice", channel="voice", tenant_id=directory.tenant_for(form["To"]),
        sender=form["From"], recipient=form["To"], text=form.get("SpeechResult", ""),
        message_id=f"{form['CallSid']}:{form.get('SequenceNumber', form.get('Confidence', '0'))}",
        thread_ref=form["CallSid"], raw=dict(form))


def twiml_gather(prompt: str, action: str, audio_url: str | None = None) -> str:
    say = f"<Play>{escape(audio_url)}</Play>" if audio_url else \
        f'<Say voice="Polly.Amy">{escape(prompt)}</Say>'
    return (f'<?xml version="1.0" encoding="UTF-8"?><Response>'
            f'<Gather input="speech" action="{escape(action)}" speechTimeout="auto" '
            f'language="en-GB">{say}</Gather>'
            f'<Say voice="Polly.Amy">Sorry, I did not catch that. Goodbye.</Say></Response>')


def twiml_say_and_hangup(text: str, audio_url: str | None = None) -> str:
    body = f"<Play>{escape(audio_url)}</Play>" if audio_url else \
        f'<Say voice="Polly.Amy">{escape(text)}</Say>'
    return f'<?xml version="1.0" encoding="UTF-8"?><Response>{body}<Hangup/></Response>'


EMPTY_TWIML = '<?xml version="1.0" encoding="UTF-8"?><Response></Response>'


class TwilioSender:
    def __init__(self, account_sid: str, auth_token: str, client: httpx.Client | None = None):
        self.sid, self.token = account_sid, auth_token
        self.http = client or httpx.Client(timeout=10)

    def send(self, *, to: str, text: str, from_: str, thread_ref: str | None = None) -> str:
        if from_.startswith("whatsapp:") and not to.startswith("whatsapp:"):
            to = f"whatsapp:{to}"
        r = self.http.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{self.sid}/Messages.json",
            auth=(self.sid, self.token), data={"To": to, "From": from_, "Body": text[:1600]})
        r.raise_for_status()
        return str(r.json()["sid"])
