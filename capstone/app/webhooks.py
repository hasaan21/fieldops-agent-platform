"""Channel webhooks. Validate, normalise, enqueue, and return fast (Twilio times out at 15s).

Voice is the exception: the caller is waiting on the line, so the agent runs synchronously.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, Response

from capstone.core.config import get_settings
from capstone.workers.channels import email as email_ch
from capstone.workers.channels import twilio as tw
from capstone.workers.channels.base import InboundMessage, NumberDirectory
from capstone.workers.channels.voice import ElevenLabsTTS, TTSCache

router = APIRouter(prefix="/webhooks", tags=["channels"])
directory = NumberDirectory()
tts_cache = TTSCache()
XML = "application/xml"


def _enqueue(msg: InboundMessage, background: BackgroundTasks) -> None:
    payload = msg.model_dump()
    try:
        from capstone.workers.celery_app import app as celery_app

        celery_app.send_task("fieldops.process_inbound", args=[payload], retry=False)
    except Exception:  # no broker (local dev): process in-process after responding
        from capstone.workers.celery_app import process

        background.add_task(process, payload)


async def _twilio_form(request: Request) -> dict[str, str]:
    form = {k: str(v) for k, v in (await request.form()).items()}
    token = get_settings().twilio_auth_token
    if token and not tw.valid_signature(token, str(request.url), form,
                                        request.headers.get("X-Twilio-Signature", "")):
        raise HTTPException(403, "invalid Twilio signature")
    return form


@router.post("/twilio/messages")
async def twilio_messages(request: Request, background: BackgroundTasks) -> Response:
    form = await _twilio_form(request)
    try:
        msg = tw.parse_message(form, directory)
    except LookupError as e:
        raise HTTPException(404, str(e)) from e
    _enqueue(msg, background)
    return Response(tw.EMPTY_TWIML, media_type=XML)


@router.post("/twilio/voice")
async def twilio_voice(request: Request) -> Response:
    form = await _twilio_form(request)
    tenant = directory.tenant_for(form["To"])
    from capstone.app.runtime import get_runtime

    name = get_runtime().crm.tenant(tenant).name
    greeting = f"Thanks for calling {name}. Briefly, how can we help today?"
    return Response(tw.twiml_gather(greeting, "/webhooks/twilio/voice/gather",
                                    _tts_url(request, greeting)), media_type=XML)


@router.post("/twilio/voice/gather")
async def twilio_voice_gather(request: Request) -> Response:
    form = await _twilio_form(request)
    msg = tw.parse_voice_speech(form, directory)
    from capstone.app.runtime import get_runtime

    result = get_runtime().handle_message(msg.tenant_id, "voice", msg.sender, msg.text,
                                          message_id=msg.message_id)
    if result["status"] == "pending_review":
        text = "Thanks. We've logged your request and will text you to confirm a time."
        return Response(tw.twiml_say_and_hangup(text, _tts_url(request, text)), media_type=XML)
    text = result.get("reply") or "Thanks for calling."
    if result.get("outcome") == "clarify":
        return Response(tw.twiml_gather(text, "/webhooks/twilio/voice/gather",
                                        _tts_url(request, text)), media_type=XML)
    return Response(tw.twiml_say_and_hangup(text, _tts_url(request, text)), media_type=XML)


def _tts_url(request: Request, text: str) -> str | None:
    s = get_settings()
    if not s.elevenlabs_api_key:
        return None
    try:
        key = tts_cache.put(text, ElevenLabsTTS(s.elevenlabs_api_key,
                                                s.elevenlabs_voice_id).synthesize(text))
    except Exception:
        return None  # fall back to Twilio <Say>
    return str(request.url_for("tts_audio", key=key))


@router.get("/tts/{key}.mp3", name="tts_audio")
def tts_audio(key: str) -> Response:
    audio = tts_cache.get(key)
    if audio is None:
        raise HTTPException(404)
    return Response(audio, media_type="audio/mpeg")


@router.post("/nylas")
async def nylas(request: Request, background: BackgroundTasks) -> dict[str, Any]:
    raw = await request.body()
    secret = get_settings().nylas_webhook_secret
    if secret and not email_ch.valid_nylas_signature(
            secret, raw, request.headers.get("X-Nylas-Signature", "")):
        raise HTTPException(403, "invalid Nylas signature")
    event = json.loads(raw)
    if event.get("type") != "message.created":
        return {"ignored": event.get("type")}
    msg = email_ch.parse_nylas_message(event["data"]["object"], directory)
    _enqueue(msg, background)
    return {"queued": msg.message_id}


@router.get("/nylas")
def nylas_challenge(challenge: str) -> Response:
    return Response(challenge, media_type="text/plain")
