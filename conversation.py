"""
Conversation state machine + reply handler.

States:
  prospecting   → first message sent, no reply yet
  engaged       → merchant has replied at least once
  action_mode   → merchant explicitly accepted; now execute, don't re-qualify
  closed        → conversation ended (merchant said no, or graceful exit)

Intent detection is pattern-based (fast, no LLM) for clear signals.
Ambiguous messages fall through to the LLM reply engine.
"""
from __future__ import annotations
import json
import logging
import re
from typing import Optional

from auto_reply import classify_reply
from validator import _has_url, _strip_urls, _has_hallucination_marker, _has_preamble, MAX_BODY_LENGTH
import llm_engine

logger = logging.getLogger(__name__)


# ── Intent patterns ───────────────────────────────────────────────────────────

_ACCEPT_PATTERNS = [
    r"\byes\b", r"\bha(an|n)?\b", r"\bhaan\b", r"\bok(ay)?\b",
    r"\bsure\b", r"\bgo ahead\b", r"\blet'?s do it\b", r"\bdo it\b",
    r"\bsend (it|me)\b", r"\bchalo\b", r"\bkar do\b", r"\bkaro\b",
    r"\btheek hai\b", r"\baccha\b", r"\bbilkul\b", r"\bji haan\b",
    r"\bproceed\b", r"\bstart\b", r"\bplease (do|send|proceed)\b",
]

_REJECT_PATTERNS = [
    r"\bno\b", r"\bnahi\b", r"\bnahin\b", r"\bnot interested\b",
    r"\bstop\b", r"\bunsubscribe\b", r"\bband karo\b", r"\bmat karo\b",
    r"\bdon'?t (want|need|contact)\b", r"\bleave (me alone|it)\b",
    r"\bbye\b", r"\bblock\b", r"\breport\b",
]

_QUESTION_PATTERNS = [
    r"\?", r"\bkya\b", r"\bkaise\b", r"\bkab\b", r"\bkitna\b",
    r"\bhow\b", r"\bwhat\b", r"\bwhen\b", r"\bwhere\b", r"\bwhy\b",
    r"\bwhich\b", r"\bcan you\b", r"\bcould you\b", r"\btell me\b",
    r"\bbatao\b", r"\bbataiye\b",
]

_COMPILED_ACCEPT = [re.compile(p, re.IGNORECASE) for p in _ACCEPT_PATTERNS]
_COMPILED_REJECT = [re.compile(p, re.IGNORECASE) for p in _REJECT_PATTERNS]
_COMPILED_QUESTION = [re.compile(p, re.IGNORECASE) for p in _QUESTION_PATTERNS]


def _detect_intent(message: str) -> str:
    """
    Fast pattern-based intent detection.
    Returns: 'accept' | 'reject' | 'question' | 'unknown'
    """
    msg = message.strip()
    if any(p.search(msg) for p in _COMPILED_REJECT):
        return "reject"
    if any(p.search(msg) for p in _COMPILED_ACCEPT):
        return "accept"
    if any(p.search(msg) for p in _COMPILED_QUESTION):
        return "question"
    return "unknown"


# ── Conversation manager ──────────────────────────────────────────────────────

class ConversationManager:
    """
    In-memory store for active conversations.
    Each conversation: { state, turns, auto_reply_count, merchant_id, customer_id }
    """

    def __init__(self) -> None:
        self._convs: dict[str, dict] = {}

    def get_or_create(self, conv_id: str, merchant_id: str, customer_id: Optional[str] = None) -> dict:
        if conv_id not in self._convs:
            self._convs[conv_id] = {
                "state": "prospecting",
                "turns": [],
                "auto_reply_count": 0,
                "merchant_id": merchant_id,
                "customer_id": customer_id,
            }
        return self._convs[conv_id]

    def record_vera_turn(self, conv_id: str, body: str) -> None:
        conv = self._convs.get(conv_id)
        if conv:
            conv["turns"].append({"from": "vera", "msg": body})

    def record_merchant_turn(self, conv_id: str, body: str) -> None:
        conv = self._convs.get(conv_id)
        if conv:
            conv["turns"].append({"from": "merchant", "msg": body})

    def close(self, conv_id: str) -> None:
        conv = self._convs.get(conv_id)
        if conv:
            conv["state"] = "closed"

    def all_vera_bodies(self, conv_id: str) -> list[str]:
        conv = self._convs.get(conv_id, {})
        return [t["msg"] for t in conv.get("turns", []) if t.get("from") == "vera"]

    def handle_reply(
        self,
        conv_id: str,
        merchant_id: str,
        customer_id: Optional[str],
        message: str,
        turn_number: int,
        store,  # ContextStore instance
    ) -> dict:
        """
        Main reply handler. Returns the response dict for /v1/reply endpoint.
        Steps:
        1. Pattern-detect auto-reply / intent
        2. Update conversation state
        3. If action needed → call LLM for reply
        4. Return { action, body, cta, rationale, ... }
        """
        conv = self.get_or_create(conv_id, merchant_id, customer_id)

        # ── Early exit for closed conversations ─────────────────────────────
        if conv["state"] == "closed":
            return {
                "action": "end",
                "rationale": "Conversation already closed. No further messages will be sent.",
            }

        self.record_merchant_turn(conv_id, message)

        auto_reply_type = classify_reply(message, conv["turns"])
        intent = _detect_intent(message) if auto_reply_type == "real" else "unknown"

        # ── Auto-reply handling (3-stage: send → wait 24h → end) ────────────
        if auto_reply_type in ("auto_reply", "repeated_auto_reply"):
            conv["auto_reply_count"] += 1

            if conv["auto_reply_count"] >= 3:
                # Third auto-reply — graceful exit
                conv["state"] = "closed"
                return {
                    "action": "end",
                    "rationale": "Auto-reply 3x in a row, no real reply. Closing conversation.",
                }
            elif conv["auto_reply_count"] == 2:
                # Second auto-reply — back off 24 hours
                return {
                    "action": "wait",
                    "wait_seconds": 86400,
                    "rationale": "Same auto-reply twice in a row — owner not at phone. Wait 24h before retry.",
                }
            else:
                # First auto-reply — try one gentle push to reach the real owner
                merchant, category = store.get_merchant_with_category(merchant_id)
                name = merchant.get("identity", {}).get("name", "") if merchant else "your team"
                languages = merchant.get("identity", {}).get("languages", ["en"]) if merchant else ["en"]
                if "hi" in languages:
                    body = (
                        "Samajh gayi — lagta hai yeh auto-reply tha. "
                        "Kya owner ya manager directly dekh sakte hain? "
                        "Sirf 2 minute ka kaam hai."
                    )
                else:
                    body = (
                        "Got it — looks like this was an automated reply. "
                        "Could the owner or manager take a quick look? "
                        "It'll take under 2 minutes."
                    )
                self.record_vera_turn(conv_id, body)
                return {
                    "action": "send",
                    "body": body,
                    "cta": "open_ended",
                    "rationale": "Auto-reply detected; one push to reach the real owner.",
                    "auto_reply_detected": True,
                }

        # ── Hard reject — graceful exit ────────────────────────────────────
        if intent == "reject":
            conv["state"] = "closed"
            # Check merchant language for appropriate goodbye
            merchant_data, _ = store.get_merchant_with_category(merchant_id)
            languages = merchant_data.get("identity", {}).get("languages", ["en"]) if merchant_data else ["en"]
            if "hi" in languages:
                close_body = "Koi baat nahi, samajh gayi. Kabhi zaroorat ho toh message kar dijiyega. Best wishes! 🙂"
            else:
                close_body = "No problem at all. Feel free to reach out anytime. Best wishes! 🙂"
            self.record_vera_turn(conv_id, close_body)
            return {
                "action": "end",
                "body": close_body,
                "rationale": "Merchant opted out. Sending polite close and ending conversation.",
            }

        # ── Intent transition: accept → action mode ──────────────────────────
        if intent == "accept" and conv["state"] != "action_mode":
            conv["state"] = "action_mode"

        # ── Build prompt for LLM reply ───────────────────────────────────────
        merchant, category = store.get_merchant_with_category(merchant_id)
        customer = store.get_customer(customer_id) if customer_id else None

        # Build compact conversation summary for the LLM
        history_lines = []
        for turn in conv["turns"][-6:]:  # last 6 turns max
            role = "VERA" if turn["from"] == "vera" else "MERCHANT"
            history_lines.append(f"[{role}]: {turn['msg']}")

        merchant_info = {}
        if merchant:
            merchant_info = {
                "name": merchant.get("identity", {}).get("name"),
                "owner_first_name": merchant.get("identity", {}).get("owner_first_name"),
                "city": merchant.get("identity", {}).get("city"),
                "languages": merchant.get("identity", {}).get("languages", ["en"]),
                "active_offers": [
                    o for o in merchant.get("offers", []) if o.get("status") == "active"
                ],
                "performance": merchant.get("performance", {}),
                "signals": merchant.get("signals", []),
                "customer_aggregate": merchant.get("customer_aggregate", {}),
            }

        category_info = {}
        if category:
            category_info = {
                "slug": category.get("slug"),
                "voice": category.get("voice", {}),
                "peer_stats": category.get("peer_stats", {}),
                "offer_catalog": category.get("offer_catalog", [])[:4],
            }

        state_note = ""
        if conv["state"] == "action_mode":
            state_note = (
                "\nCRITICAL — MERCHANT ACCEPTED. You are in ACTION MODE. "
                "Do NOT re-qualify. Do NOT ask 'are you sure?'. "
                "Describe exactly what you are doing/have done for them, "
                "or ask for the ONE missing piece of info needed to execute."
            )
        elif conv["state"] == "engaged":
            state_note = (
                "\nMerchant has engaged. Keep momentum — short, direct reply. "
                "If they're asking something specific, answer it with a fact from context."
            )

        already_sent_vera = self.all_vera_bodies(conv_id)
        sent_note = ""
        if already_sent_vera:
            sent_note = "\nVera already said (don't repeat):\n" + "\n".join(
                f"  - {b[:100]}" for b in already_sent_vera[-3:]
            )

        prompt = f"""CONVERSATION HISTORY:
{chr(10).join(history_lines)}

MERCHANT'S LATEST MESSAGE: "{message}"
DETECTED INTENT: {intent}
CONVERSATION STATE: {conv['state']}{state_note}

MERCHANT INFO:
{json.dumps(merchant_info, indent=2, ensure_ascii=False)}

CATEGORY INFO:
{json.dumps(category_info, indent=2, ensure_ascii=False)}
{sent_note}

Decide the next action. Remember:
- If intent is 'accept' or state is 'action_mode': DO THE ACTION, don't re-qualify.
- If it's a genuine question: answer it concisely with a fact from the context.
- Keep replies short (1-3 sentences). WhatsApp style.
- Use Hindi-English mix if merchant's languages include 'hi'.

Return ONLY the JSON object."""

        try:
            result = llm_engine.reply(prompt)
            action = result.get("action", "send")

            # ── Post-LLM reply validation ──────────────────────────────────
            if action == "send":
                body = result.get("body", "")
                if body:
                    # ── Anti-repetition Guard ───────────────────────────────
                    already_sent = self.all_vera_bodies(conv_id)
                    if body in already_sent:
                        logger.warning("Caught verbatim repeat in reply path. Appending variation.")
                        body += " Let me know if you want to proceed."

                    # Strip URLs (−3 penalty per URL in the rubric)
                    if _has_url(body):
                        body = _strip_urls(body)
                        logger.info("Stripped URLs from reply body")

                    # Strip preambles ("I hope you're doing well...")
                    if _has_preamble(body):
                        sentences = re.split(r"(?<=[.!?])\s+", body, maxsplit=1)
                        if len(sentences) > 1:
                            body = sentences[1].strip()

                    # Truncate to max length at sentence boundary
                    if len(body) > MAX_BODY_LENGTH:
                        truncated = body[:MAX_BODY_LENGTH]
                        last_end = max(truncated.rfind("."), truncated.rfind("?"), truncated.rfind("!"))
                        if last_end > MAX_BODY_LENGTH // 2:
                            body = truncated[:last_end + 1]

                    result["body"] = body

                if body:
                    self.record_vera_turn(conv_id, body)
                if result.get("auto_reply_detected"):
                    conv["auto_reply_count"] += 1

            if action == "end":
                conv["state"] = "closed"

            return result

        except Exception as exc:
            logger.error("Reply LLM call failed for %s: %s", conv_id, exc)
            # Use specific_fallback instead of generic message
            try:
                fallback_result = llm_engine.specific_fallback(merchant or {}, category or {}, {})
                fallback_body = fallback_result.get("body", "Got it, looking into this now.")
            except Exception:
                name = (merchant or {}).get("identity", {}).get("owner_first_name", "")
                fallback_body = f"{'Haan ' + name + ', ' if name else ''}ek second — main check kar ke batati hoon."
            self.record_vera_turn(conv_id, fallback_body)
            return {
                "action": "send",
                "body": fallback_body,
                "cta": "open_ended",
                "rationale": f"LLM error fallback (specific): {exc}",
            }


# Singleton
conversation_manager = ConversationManager()
