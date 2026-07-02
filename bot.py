"""
Vera Bot — FastAPI server for the magicpin AI Challenge.

Implements all 5 required endpoints:
  GET  /v1/healthz    — liveness probe
  GET  /v1/metadata   — bot identity
  POST /v1/context    — receive context pushes (idempotent by scope+version)
  POST /v1/tick       — periodic wake-up; bot initiates proactive messages
  POST /v1/reply      — receive merchant/customer replies; bot responds

Run:
  uvicorn bot:app --host 0.0.0.0 --port 8080 --reload
"""
from __future__ import annotations
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Load .env before any other import that reads env vars
load_dotenv()

from contexts import store
from composer import select_and_compose
from conversation import conversation_manager

from submission_cache import _load_cache
_load_cache()  # Pre-load cache on startup

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vera.bot")

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Vera Bot", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the demo UI
public_dir = Path(__file__).parent / "public"
if public_dir.exists():
    app.mount("/demo", StaticFiles(directory=str(public_dir), html=True), name="demo")

    @app.get("/")
    async def root():
        return FileResponse(public_dir / "index.html")

START_TIME = time.time()

# Track suppression keys that have already been used this session
sent_suppression_keys: set[str] = set()


# ── Pydantic models ───────────────────────────────────────────────────────────

class ContextBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str


class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/v1/healthz")
async def healthz():
    counts = store.counts()
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": counts,
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Vera Intelligence",
        "team_members": ["Santosh Malana"],
        "model": f"{os.getenv('GEMINI3_MODEL', 'gemini-3-flash-preview')} (primary) + {os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')} (fallback)",
        "approach": (
            "Trigger-kind router with per-kind prompt variants, "
            "pattern-based auto-reply detection, conversation state machine "
            "(prospecting→engaged→action_mode→closed), "
            "post-LLM output validation, and priority-scored trigger selection."
        ),
        "contact_email": "santosh24x@gmail.com",
        "version": "1.0.0",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/v1/context")
async def push_context(body: ContextBody, background_tasks: BackgroundTasks):
    valid_scopes = {"category", "merchant", "customer", "trigger"}
    if body.scope not in valid_scopes:
        return JSONResponse(
            status_code=400,
            content={"accepted": False, "reason": "invalid_scope", "details": f"Must be one of {valid_scopes}"},
        )

    result = store.store(body.scope, body.context_id, body.version, body.payload)

    if not result["accepted"]:
        return JSONResponse(
            status_code=409,
            content={
                "accepted": False,
                "reason": "stale_version",
                "current_version": result["current_version"],
            },
        )

    if body.scope == "trigger":
        from generate_cache import prewarm_trigger
        background_tasks.add_task(prewarm_trigger, body.payload)

    logger.info("Stored %s/%s v%d", body.scope, body.context_id, body.version)
    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/v1/tick")
async def tick(body: TickBody):
    logger.info("Tick at %s — %d triggers available", body.now, len(body.available_triggers))

    if not body.available_triggers:
        return {"actions": []}

    try:
        actions = select_and_compose(
            available_trigger_ids=body.available_triggers,
            store=store,
            sent_suppression_keys=sent_suppression_keys,
            conversations=conversation_manager._convs,
        )
    except Exception as exc:
        logger.error("Tick compose error: %s", exc, exc_info=True)
        return {"actions": []}

    # Record suppression keys and vera turns for all actions we're sending
    for action in actions:
        sk = action.get("suppression_key", "")
        if sk:
            sent_suppression_keys.add(sk)

        conv_id = action.get("conversation_id", "")
        merchant_id = action.get("merchant_id", "")
        customer_id = action.get("customer_id")
        body_text = action.get("body", "")

        if conv_id and body_text and body_text.strip():
            conversation_manager.get_or_create(conv_id, merchant_id, customer_id)
            conversation_manager.record_vera_turn(conv_id, body_text.strip())

    logger.info("Tick returning %d actions", len(actions))
    return {"actions": actions}


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    logger.info(
        "Reply in conv %s from %s (turn %d): %s",
        body.conversation_id,
        body.from_role,
        body.turn_number,
        body.message[:80],
    )

    result = conversation_manager.handle_reply(
        conv_id=body.conversation_id,
        merchant_id=body.merchant_id or "",
        customer_id=body.customer_id,
        from_role=body.from_role,
        message=body.message,
        turn_number=body.turn_number,
        store=store,
    )

    # Strip internal fields before returning to judge
    result.pop("auto_reply_detected", None)
    
    # Ensure a body key is always returned to prevent judge errors
    if "body" not in result and result.get("action") in ("wait", "end"):
        result["body"] = ""
        
    return result


# ── Optional teardown endpoint ────────────────────────────────────────────────

@app.post("/v1/teardown")
async def teardown():
    store._store.clear()
    conversation_manager._convs.clear()
    conversation_manager._merchant_auto_replies.clear()
    sent_suppression_keys.clear()
    
    import submission_cache
    submission_cache._cache.clear()
    
    logger.info("State wiped on teardown.")
    return {"status": "wiped"}


# ── Dispatch endpoint (Trojan Horse execution) ────────────────────────────────

class DispatchBody(BaseModel):
    merchant_id: str
    message_body: str
    target_customer_ids: list[str] = []
    conversation_id: str = ""

@app.post("/v1/dispatch")
async def dispatch_customer_message(body: DispatchBody):
    """
    Execute a pre-drafted customer-facing message on behalf of the merchant.
    This is the backend for the zero-friction 'Reply Send' Trojan Horse pattern.
    
    In production, this would push to WhatsApp Business API.
    For the challenge, we log the dispatch and return success.
    """
    target_count = len(body.target_customer_ids) or 1
    logger.info(
        "DISPATCH: merchant=%s sending to %d customers. Body: %s",
        body.merchant_id,
        target_count,
        body.message_body[:100],
    )
    
    # Record the dispatch in conversation state
    if body.conversation_id:
        conversation_manager.record_dispatch(
            conv_id=body.conversation_id,
            merchant_id=body.merchant_id,
            dispatched_body=body.message_body,
            target_count=target_count,
        )
    
    return {
        "status": "dispatched",
        "merchant_id": body.merchant_id,
        "recipients": target_count,
        "message_preview": body.message_body[:80],
    }


# ── Global exception handler ──────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": str(exc)},
    )
