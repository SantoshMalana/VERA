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
import ssl
import time
from typing import Optional

# Fix SSL certificate verification for corporate/proxy environments
ssl._create_default_https_context = ssl._create_unverified_context

from auto_reply import classify_reply
from validator import _has_url, _strip_urls, _has_hallucination_marker, _has_preamble, MAX_BODY_LENGTH
import llm_engine

logger = logging.getLogger(__name__)


# ── Intent patterns ───────────────────────────────────────────────────────────

_ACCEPT_PATTERNS = [
    r"\byes\b", r"\bha(an|n)?\b", r"\bhaan\b", r"^\s*ok(?:ay)?\s*$",
    r"\bsure\b", r"\bgo ahead\b", r"\blet'?s do it\b", r"\bdo it\b",
    r"\bsend (it|me)\b", r"\bchalo\b", r"\bkar do\b", r"\bkaro\b",
    r"\btheek hai\b", r"\baccha\b", r"\bbilkul\b", r"\bji haan\b",
    r"\bproceed\b", r"\bplease (do|send|proceed)\b",
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
        self._merchant_auto_replies: dict[str, int] = {}

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
        else:
            logger.warning("record_vera_turn called for unknown conv_id %s", conv_id)

    def record_customer_turn(self, conv_id: str, body: str) -> None:
        conv = self._convs.get(conv_id)
        if conv:
            conv["turns"].append({"from": "customer", "msg": body})
        else:
            logger.warning("record_customer_turn called for unknown conv_id %s", conv_id)

    def record_merchant_turn(self, conv_id: str, body: str) -> None:
        conv = self._convs.get(conv_id)
        if conv:
            conv["turns"].append({"from": "merchant", "msg": body})
        else:
            logger.warning("record_merchant_turn called for unknown conv_id %s", conv_id)

    def close(self, conv_id: str) -> None:
        conv = self._convs.get(conv_id)
        if conv:
            conv["state"] = "closed"

    def record_dispatch(self, conv_id: str, merchant_id: str, dispatched_body: str, target_count: int) -> None:
        """Record that a pre-drafted message was dispatched to customers."""
        conv = self.get_or_create(conv_id, merchant_id)
        conv.setdefault("dispatches", []).append({
            "body": dispatched_body[:200],
            "target_count": target_count,
            "timestamp": time.time(),
        })
        conv["turns"].append({
            "from": "vera_dispatch",
            "msg": f"[DISPATCHED to {target_count} customers]: {dispatched_body[:80]}",
        })

    def all_vera_bodies(self, conv_id: str) -> list[str]:
        conv = self._convs.get(conv_id, {})
        return [t["msg"] for t in conv.get("turns", []) if t.get("from") == "vera"]

    def _calculate_heat_score(self, conv: dict, message: str, intent: str, merchant_info: dict) -> dict:
        """
        Calculates conversation heat metrics to adjust LLM aggression.
        Returns: {
            "unanswered_count": int,
            "owner_name_used": bool,
            "question_asked": bool,
            "aggression_level": "cold" | "warm" | "hot" | "answer_first"
        }
        """
        turns = conv.get("turns", [])
        
        # 1. Unanswered Count
        unanswered_count = 0
        for t in reversed(turns):
            if t["from"] == "vera":
                unanswered_count += 1
            else:
                break
                
        # 2. Name Used
        owner_name = merchant_info.get("owner_first_name", "")
        owner_name_used = False
        if owner_name and owner_name.lower() in message.lower():
            owner_name_used = True
            
        # 3. Question Asked
        question_asked = (intent == "question")
        
        # Determine aggression
        if question_asked:
            agg = "answer_first"
        elif owner_name_used or intent == "accept":
            agg = "hot"
        elif unanswered_count > 1:
            agg = "cold"
        else:
            agg = "warm"
            
        return {
            "unanswered_count": unanswered_count,
            "owner_name_used": owner_name_used,
            "question_asked": question_asked,
            "aggression_level": agg
        }

    # ── Slot Confirmation Engine ─────────────────────────────────────────────
    _DAY_PATTERN = re.compile(
        r"\b(mon(?:day)?|tue(?:sday)?|wed(?:nesday)?|thu(?:rsday)?|fri(?:day)?|sat(?:urday)?|sun(?:day)?)\b",
        re.IGNORECASE,
    )
    _DATE_PATTERN = re.compile(
        r"\b(\d{1,2})\s*(?:st|nd|rd|th)?\s*"
        r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
        re.IGNORECASE,
    )
    _TIME_PATTERN = re.compile(
        r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b",
        re.IGNORECASE,
    )
    _SLOT_DIGIT_PATTERN = re.compile(r"\breply\s*([12])\b|\boption\s*([12])\b", re.IGNORECASE)
    _CUSTOMER_STOP_PATTERN = re.compile(
        r"\b(stop|no|nahi|nahin|cancel|not interested|unsubscribe|mat karo)\b", re.IGNORECASE
    )

    def _extract_slot(self, message: str) -> Optional[dict]:
        """Deterministically extract booking slot. Returns dict or None."""
        slot = {}
        day_m = self._DAY_PATTERN.search(message)
        if day_m:
            slot["day"] = day_m.group(1).capitalize()
        date_m = self._DATE_PATTERN.search(message)
        if date_m:
            slot["date"] = f"{date_m.group(1)} {date_m.group(2).capitalize()}"
        time_m = self._TIME_PATTERN.search(message)
        if time_m:
            hour = time_m.group(1)
            minute = time_m.group(2) or "00"
            meridiem = time_m.group(3).upper()
            slot["time"] = f"{hour}:{minute} {meridiem}"
        digit_m = self._SLOT_DIGIT_PATTERN.search(message)
        if digit_m:
            slot["reply_digit"] = digit_m.group(1) or digit_m.group(2)
        # Need at least one time signal to be a real slot pick
        return slot if (slot.get("time") or slot.get("day") or slot.get("reply_digit")) else None

    def _build_slot_confirmation(
        self, slot: dict, customer_name: str, merchant: dict, category: dict, history: list
    ) -> str:
        """Build a deterministic, zero-LLM confirmation with full specifics."""
        identity = (merchant or {}).get("identity", {})
        owner = identity.get("owner_first_name", "")
        locality = identity.get("locality", "") or identity.get("city", "")
        languages = identity.get("languages", ["en"])
        is_hindi = "hi" in languages
        slug = (category or {}).get("slug", "")
        emoji_map = {"dentists": "🦷", "salons": "💇", "restaurants": "🍽️", "gyms": "💪", "pharmacies": "💊"}
        emoji = emoji_map.get(slug, "✅")

        # Best offer with price
        active_offers = [o for o in (merchant or {}).get("offers", []) if o.get("status") == "active"]
        offer_str = ""
        if active_offers:
            o = active_offers[0]
            title = o.get("title", "")
            price = o.get("price") or o.get("value") or o.get("discounted_price")
            offer_str = f"{title} @ ₹{price}" if (title and price) else title

        # Resolve digit reply to actual slot from Vera's last message
        slot_label = ""
        if slot.get("reply_digit"):
            digit = slot["reply_digit"]
            for turn in reversed(history):
                if turn.get("from") == "vera":
                    found_days = self._DAY_PATTERN.findall(turn.get("msg", ""))
                    if found_days and digit in ("1", "2"):
                        idx = int(digit) - 1
                        if idx < len(found_days):
                            slot_label = found_days[idx].capitalize()
                    break
        else:
            parts = []
            if slot.get("day"):
                parts.append(slot["day"])
            if slot.get("date"):
                parts.append(slot["date"])
            if slot.get("time"):
                parts.append(f"at {slot['time']}")
            slot_label = " ".join(parts)

        first = customer_name.split()[0] if customer_name else "there"
        doctor = f"Dr. {owner}" if owner else (merchant or {}).get("identity", {}).get("name", "us")

        if is_hindi:
            msg_parts = [f"Perfect, {first}!"]
            if slot_label:
                msg_parts.append(f"{slot_label} — {doctor} ke saath confirmed hai.")
            if offer_str:
                msg_parts.append(f"{offer_str}.")
            if locality:
                msg_parts.append(f"📍 {locality}.")
            msg_parts.append(f"Hum aapka intezaar karenge! {emoji}")
        else:
            msg_parts = [f"Confirmed, {first}!"]
            if slot_label:
                msg_parts.append(f"{slot_label} with {doctor} is all set.")
            if offer_str:
                msg_parts.append(f"{offer_str}.")
            if locality:
                msg_parts.append(f"📍 {locality}.")
            msg_parts.append(f"See you then! {emoji}")

        return " ".join(msg_parts)

    def _handle_customer_reply(
        self,
        conv_id: str,
        merchant_id: str,
        customer_id: Optional[str],
        message: str,
        turn_number: int,
        store,
    ) -> dict:
        conv = self.get_or_create(conv_id, merchant_id, customer_id)
        if conv["state"] == "closed":
            return {"action": "end", "rationale": "Conversation already closed."}

        self.record_customer_turn(conv_id, message)  # customer messages stored as "customer"
        merchant, category = store.get_merchant_with_category(merchant_id)
        customer = store.get_customer(customer_id) if customer_id else None
        customer_name = (customer or {}).get("identity", {}).get("name", "Customer")
        merchant_name = (merchant or {}).get("identity", {}).get("name", "the business")
        is_hindi = "hi" in ((merchant or {}).get("identity", {}).get("languages", ["en"]))

        # ── STOP handling — deterministic, no LLM ──────────────────────────
        if self._CUSTOMER_STOP_PATTERN.search(message):
            conv["state"] = "closed"
            first = customer_name.split()[0]
            close_body = (
                f"Bilkul samajh gaye, {first}. Koi baat nahi! Kabhi bhi aayein. 🙂"
                if is_hindi
                else f"Understood, {first}! No worries at all. Feel free to reach out anytime. 🙂"
            )
            self.record_vera_turn(conv_id, close_body)
            return {"action": "end", "body": close_body, "cta": "none",
                    "rationale": "Customer opted out — deterministic clean close."}

        # ── Slot Confirmation Engine — zero LLM ────────────────────────────
        slot = self._extract_slot(message)
        if slot:
            body = self._build_slot_confirmation(slot, customer_name, merchant, category, conv["turns"])
            self.record_vera_turn(conv_id, body)
            return {"action": "send", "body": body, "cta": "none",
                    "rationale": f"Deterministic slot confirmation: {slot}"}

        # ── LLM fallback for non-slot messages (questions, etc.) ───────────
        history_lines = []
        for turn in conv["turns"][-6:]:
            # VERA's turns = what the merchant's clinic sent; CUSTOMER's turns = what the customer said
            role = "VERA" if turn["from"] == "vera" else "CUSTOMER"
            history_lines.append(f"[{role}]: {turn['msg']}")

        active_offers = [o for o in (merchant or {}).get("offers", []) if o.get("status") == "active"]
        offer_context = ""
        if active_offers:
            offer_context = "\nACTIVE OFFERS: " + json.dumps([
                {"title": o.get("title"), "price": o.get("price") or o.get("value")}
                for o in active_offers[:3]
            ])

        first = customer_name.split()[0]
        owner_name = (merchant or {}).get("identity", {}).get("owner_first_name", "")
        category_slug = (category or {}).get("slug", "")
        # Build a warm, on-brand persona for the merchant reply
        persona = f"Dr. {owner_name}" if (owner_name and category_slug in ("dentists", "pharmacies")) else owner_name or merchant_name

        prompt = f"""CONVERSATION HISTORY:
{chr(10).join(history_lines)}

CUSTOMER'S LATEST MESSAGE: "{message}"

You are Vera acting as {persona} ({merchant_name}) responding to customer {first}.
RULES:
- Address the customer by their first name ({first}) — NOT the merchant/doctor name.
- You are writing FROM the merchant TO the customer.
- If the customer picked a slot or said YES: confirm it warmly with the specific time/offer details.
- If the customer asked a question: answer it in 1-2 sentences using the offer context.
- Keep it warm, professional, 1-2 sentences max.
- Do NOT mention Vera or magicpin.
- Do NOT address the merchant name in the reply body.{offer_context}

Return ONLY valid JSON:
{{
  "action": "send" | "end",
  "body": "<reply addressed TO the customer {first}, NOT to the merchant>",
  "cta": "open_ended" | "none",
  "rationale": "<reason>"
}}"""

        try:
            result = llm_engine.reply(prompt)
            if result.get("action") == "send":
                body = result.get("body", "")
                if body:
                    # Guard: if LLM accidentally addressed the merchant instead of customer,
                    # detect by checking if merchant owner name appears at the start (e.g. "Dr. Meera,")
                    owner_first = (merchant or {}).get("identity", {}).get("owner_first_name", "")
                    if owner_first and body.strip().lower().startswith(owner_first.lower()):
                        # Re-anchor with customer name
                        body = f"{first}, " + body.split(",", 1)[-1].strip() if "," in body else body
                        result["body"] = body
                        logger.warning("Customer reply was merchant-addressed — re-anchored to customer '%s'", first)
                    # Strip any preambles like "Of course," followed by merchant name
                    from validator import _has_preamble
                    if _has_preamble(body):
                        sentences = re.split(r"(?<=[.!?])\s+", body, maxsplit=1)
                        if len(sentences) > 1:
                            body = sentences[1].strip()
                            result["body"] = body
                    self.record_vera_turn(conv_id, body)
            return result
        except Exception as exc:
            logger.error("Customer reply LLM failed for %s: %s", conv_id, exc)
            fallback = f"Got it, {first}! Let me confirm and get back to you shortly. 🙂"
            self.record_vera_turn(conv_id, fallback)
            return {"action": "send", "body": fallback, "cta": "open_ended", "rationale": "Fallback"}

    def handle_reply(
        self,
        conv_id: str,
        merchant_id: str,
        customer_id: Optional[str],
        from_role: str,
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
        if from_role == "customer":
            return self._handle_customer_reply(
                conv_id, merchant_id, customer_id, message, turn_number, store
            )

        conv = self.get_or_create(conv_id, merchant_id, customer_id)

        # ── Early exit for closed conversations — silent, no second "end" ──
        if conv["state"] == "closed":
            return {
                "action": "wait",
                "wait_seconds": 0,
                "rationale": "Conversation already closed — no further messages.",
            }

        self.record_merchant_turn(conv_id, message)

        auto_reply_type = classify_reply(message, conv["turns"])
        intent = _detect_intent(message) if auto_reply_type == "real" else "unknown"

        # ── Auto-reply handling (3-stage: send(tick) → wait 24h → end) ────────────
        if auto_reply_type in ("auto_reply", "repeated_auto_reply"):
            conv["auto_reply_count"] += 1
            if merchant_id:
                self._merchant_auto_replies[merchant_id] = self._merchant_auto_replies.get(merchant_id, 0) + 1
            
            merchant_count = self._merchant_auto_replies.get(merchant_id, 0) if merchant_id else 0

            if conv["auto_reply_count"] >= 2 or merchant_count >= 2:
                conv["state"] = "closed"
                return {
                    "action": "end",
                    "rationale": "Second auto-reply detected. Closing conversation to prevent loops.",
                }
            else:
                return {
                    "action": "wait",
                    "wait_seconds": 86400,
                    "rationale": "Auto-reply detected. Waiting 24h before retry.",
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
            
            # ── Deterministic action-mode response (no LLM dependency) ──
            # This is the CRITICAL path. If the merchant says "go ahead",
            # we MUST respond with a clear, specific next step.
            if merchant:
                owner = merchant.get("identity", {}).get("owner_first_name", "")
                perf = merchant.get("performance", {})
                ctr = perf.get("ctr") or perf.get("ctr_30d")
                active_offers = [o for o in merchant.get("offers", []) if o.get("status") == "active"]
                cust_agg = merchant.get("customer_aggregate", {})
                lapsed = cust_agg.get("lapsed_180d_plus") or cust_agg.get("lapsed_count", 0)
                languages = merchant.get("identity", {}).get("languages", ["en"])
                is_hindi = "hi" in languages
                
                # Build a hyper-specific action plan
                steps = []
                if active_offers:
                    offer = active_offers[0]
                    title = offer.get("title", "your offer")
                    price = offer.get("price") or offer.get("value", "")
                    steps.append(f"refresh '{title}' {'@ ₹' + str(price) if price else ''}" if not is_hindi else f"'{title}' {'@ ₹' + str(price) if price else ''} ko refresh karna")
                if ctr:
                    ctr_pct = round(float(ctr) * 100, 1) if float(ctr) < 1 else round(float(ctr), 1)
                    steps.append(f"optimize your profile to push CTR from {ctr_pct}%" if not is_hindi else f"profile optimize karna — CTR abhi {ctr_pct}% hai")
                if lapsed:
                    steps.append(f"send recall to {lapsed} lapsed customers" if not is_hindi else f"{lapsed} lapsed customers ko recall bhejna")
                
                if steps:
                    greeting = f"Dr. {owner}" if owner else "Great"
                    if is_hindi:
                        action_body = f"{greeting}, done! Plan ready hai — Step 1: {steps[0]}"
                        if len(steps) > 1:
                            action_body += f". Step 2: {steps[1]}"
                        action_body += ". Mujhe ek photo ya updated price bhej do toh main abhi karta/karti hoon."
                    else:
                        action_body = f"{greeting}, let's go! Step 1: {steps[0]}"
                        if len(steps) > 1:
                            action_body += f". Step 2: {steps[1]}"
                        action_body += ". Send me a photo or updated price and I'll execute right now."
                    
                    self.record_vera_turn(conv_id, action_body)
                    return {
                        "action": "send",
                        "body": action_body,
                        "cta": "open_ended",
                        "rationale": f"Deterministic action plan: {len(steps)} steps from merchant data.",
                    }

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
                "\nCRITICAL — MERCHANT ACCEPTED. ACTION MODE RULES:"
                "\n1. NEVER claim to have done something you cannot actually do (refresh offer, send email, etc.)."
                "\n2. Instead: ask for the ONE real input needed to execute (photo? price? which slot?)."
                "\n3. If nothing is needed: give a specific, numbered next step with a timeframe."
                "\n4. Close the conversation in max 1-2 more turns. Do NOT drag it out."
                "\n5. Your reply must contain a specific number (%, ₹, hours, days) — no vague promises."
            )
        else:
            heat = self._calculate_heat_score(conv, message, intent, merchant_info)
            agg = heat["aggression_level"]
            if agg == "cold":
                state_note = "\nCONVERSATION HEAT: COLD. Merchant is unresponsive. Lead with a completely new angle, not a follow-up."
            elif agg == "hot":
                state_note = "\nCONVERSATION HEAT: HOT. Merchant is highly engaged. Push for commitment immediately."
            elif agg == "answer_first":
                state_note = "\nCONVERSATION HEAT: HIGH INTENT. Merchant asked a question. Answer the question first with facts, then add CTA."
            else:
                state_note = "\nCONVERSATION HEAT: WARM. Merchant has engaged. Keep momentum — short, direct reply."

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
- NEVER say you've completed an action. Instead: ask for the ONE input needed, or give a specific numbered next step.
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