"""
Trigger-kind router + prompt builder.

Each trigger kind gets:
  - A specific angle (what to lead with)
  - Pre-computed insights injected from insights.py
  - Post-LLM validation from validator.py

Flow:
  compose_message() → build_prompt() → llm_engine.compose() → validate_and_repair()
"""
from __future__ import annotations
import json
import logging
from typing import Any, Optional

import llm_engine
from insights import derive_insights
from validator import validate_and_repair

logger = logging.getLogger(__name__)


# ── Trigger-kind strategy map ─────────────────────────────────────────────────

_KIND_STRATEGIES: dict[str, dict] = {
    "research_digest": {
        "angle": (
            "A new research/regulatory digest item arrived. Lead with the single most relevant "
            "finding for this merchant — use the exact stat, trial size, and source citation from "
            "the digest. Offer to pull the full abstract or draft a patient-education message. "
            "LEVER: CURIOSITY + EFFORT EXTERNALIZATION."
        ),
        "cta_hint": "open_ended",
    },
    "research_digest_release": {
        "angle": (
            "New category research digest released. Pick the finding most relevant to THIS "
            "merchant's signals (e.g. if they have high-risk patients, cite that trial). "
            "Use the specific numbers. Offer to draft a shareable patient content piece. "
            "LEVER: SPECIFICITY + CURIOSITY."
        ),
        "cta_hint": "open_ended",
    },
    "category_research_digest_release": {
        "angle": (
            "New category research digest released. Pick the finding most relevant to THIS "
            "merchant's signals. Use specific numbers and source citation. Offer to draft "
            "a shareable patient content piece. LEVER: SPECIFICITY + CURIOSITY."
        ),
        "cta_hint": "open_ended",
    },
    "perf_spike": {
        "angle": (
            "Merchant's performance spiked — use the EXACT metric and delta from performance_delta "
            "insight. Celebrate briefly (peer tone, not promotional). Immediately offer the next "
            "action to convert visibility into bookings. LEVER: RECIPROCITY + SPECIFICITY."
        ),
        "cta_hint": "binary_yes_stop",
    },
    "perf_dip": {
        "angle": (
            "Metric dropped — state the EXACT numbers (before and after). Frame as recoverable. "
            "Offer ONE concrete action (refresh offer, add post, update photos) that directly "
            "addresses the dip. Use effort externalization: 'I can do this for you — just say YES'. "
            "LEVER: LOSS AVERSION + EFFORT EXTERNALIZATION."
        ),
        "cta_hint": "binary_yes_stop",
    },
    "milestone_reached": {
        "angle": (
            "Merchant crossed a milestone. Mention the exact number. Acknowledge peer-to-peer "
            "(not promotional). Immediately pivot to the NEXT milestone or a compounding action. "
            "LEVER: SOCIAL PROOF + CURIOSITY."
        ),
        "cta_hint": "open_ended",
    },
    "dormant_with_vera": {
        "angle": (
            "Merchant has been inactive 14+ days. Do NOT say 'you've been away'. Instead, "
            "lead with a NEW specific insight from their account derived from the insights block: "
            "CTR gap, lapsed revenue opportunity, or stale content. Make it feel like you noticed "
            "something useful, not that you're chasing them. LEVER: RECIPROCITY + SPECIFICITY."
        ),
        "cta_hint": "open_ended",
    },
    "competitor_opened": {
        "angle": (
            "A new competitor opened nearby. Use the exact distance and location from the trigger. "
            "Frame as useful market intelligence. Suggest one differentiation action. "
            "LEVER: CURIOSITY ('want to see their profile?') + LOSS AVERSION."
        ),
        "cta_hint": "binary_yes_stop",
    },
    "festival_upcoming": {
        "angle": (
            "A festival is X days away (use exact days from trigger). Pair with a SPECIFIC "
            "service+price from the merchant's active offer catalog — not a generic '% off'. "
            "Suggest running the campaign NOW to beat competitors. "
            "LEVER: URGENCY + SPECIFICITY."
        ),
        "cta_hint": "binary_yes_stop",
    },
    "weather_heatwave": {
        "angle": (
            "Extreme weather in the merchant's city. Connect it to a real category impact: "
            "salons → hair damage, restaurants → delivery surge, gyms → indoor preference. "
            "Brief timely nudge. LEVER: TIMELINESS + RECIPROCITY."
        ),
        "cta_hint": "open_ended",
    },
    "local_news_event": {
        "angle": (
            "Local news event affects the merchant's area. Name the event, explain the business "
            "impact directly (foot traffic, demand shift), offer a concrete action. "
            "LEVER: RECIPROCITY + TIMELINESS."
        ),
        "cta_hint": "binary_yes_stop",
    },
    "regulation_change": {
        "angle": (
            "Regulatory/compliance change. Lead with the specific regulation name and effective "
            "date from the payload. Explain the direct impact on this merchant's practice. "
            "Offer to help with compliance. Clinical/peer tone, never alarmist. "
            "LEVER: RECIPROCITY + SPECIFICITY."
        ),
        "cta_hint": "open_ended",
    },
    "category_trend_movement": {
        "angle": (
            "Category trend is moving. Lead with the specific trend signal and % change from "
            "the top_trend insight. Connect it to an offer/service the merchant already has. "
            "LEVER: SOCIAL PROOF + LOSS AVERSION."
        ),
        "cta_hint": "binary_yes_stop",
    },
    "review_theme_emerged": {
        "angle": (
            "Multiple reviews mention the same theme. Name the exact theme. "
            "Positive theme → amplify it in a post. Negative theme → offer a concrete fix. "
            "LEVER: RECIPROCITY + SPECIFICITY."
        ),
        "cta_hint": "binary_yes_stop",
    },
    "recall_due": {
        "angle": (
            "Customer is due for their recall appointment. This message goes FROM THE MERCHANT "
            "TO THEIR CUSTOMER. Use the customer's name, exact time since last visit, their "
            "preferred slot times, and a specific ₹-priced offer. Offer 2 concrete time slots. "
            "LEVER: SPECIFICITY + EFFORT EXTERNALIZATION."
        ),
        "cta_hint": "open_ended",
    },
    "customer_lapsed_soft": {
        "angle": (
            "Customer hasn't returned in 3-6 months. Message from merchant to customer. "
            "Lead with the time gap and their last service. Offer a specific re-engagement deal "
            "with a ₹ price. Warm but not desperate. LEVER: RECIPROCITY + LOSS AVERSION."
        ),
        "cta_hint": "binary_yes_stop",
    },
    "customer_lapsed_hard": {
        "angle": (
            "Customer gone 6+ months. Message from merchant to customer. "
            "Time-sensitive specific offer with a ₹ price. Warm, not needy. "
            "LEVER: LOSS AVERSION + SPECIFICITY."
        ),
        "cta_hint": "binary_yes_stop",
    },
    "appointment_tomorrow": {
        "angle": (
            "Appointment reminder on behalf of merchant. Include: merchant name, time, address "
            "or landmark. No CTA needed. Friendly and specific. "
            "LEVER: EFFORT EXTERNALIZATION (Vera handles the reminder)."
        ),
        "cta_hint": "none",
    },
    "scheduled_recurring": {
        "angle": (
            "Curiosity/knowledge-driven nudge — not a functional reminder. Pick the MOST "
            "interesting insight from the insights block: CTR gap, lapsed revenue, seasonal beat, "
            "or trend signal. Ask the merchant one genuine question. "
            "LEVER: CURIOSITY + ASKING THE MERCHANT."
        ),
        "cta_hint": "open_ended",
    },
    "ipl_match_today": {
        "angle": (
            "IPL match is happening today in the merchant's city. "
            "Lead with the exact team playing (from trigger payload). "
            "Frame as a local footfall opportunity — IPL evenings drive "
            "delivery/dine-in/salon appointments. Suggest ONE specific "
            "action: run a match-night offer. LEVER: TIMELINESS + SPECIFICITY."
        ),
        "cta_hint": "binary_yes_stop",
    },
    "renewal_due": {
        "angle": (
            "Subscription is expiring in X days (use exact days_remaining from payload). "
            "Lead with the SPECIFIC renewal amount (₹ from payload). Frame as protecting "
            "what's already working — don't let momentum die. "
            "LEVER: LOSS AVERSION + SPECIFICITY."
        ),
        "cta_hint": "binary_yes_stop",
    },
    "winback_eligible": {
        "angle": (
            "Merchant's subscription lapsed X days ago (use exact days_since_expiry). "
            "Lead with a concrete loss they've incurred: lapsed_customers_added_since_expiry "
            "customers have gone uncontacted. Frame as recoverable NOW. "
            "LEVER: LOSS AVERSION + SPECIFICITY."
        ),
        "cta_hint": "binary_yes_stop",
    },
    "supply_alert": {
        "angle": (
            "Drug/product recall alert. Lead with the exact molecule name and affected batch "
            "numbers from the payload. Clinical-peer tone — never alarmist. "
            "Offer to help identify affected stock. LEVER: RECIPROCITY + URGENCY."
        ),
        "cta_hint": "open_ended",
    },
    "chronic_refill_due": {
        "angle": (
            "Customer's chronic medications are running out (use exact stock_runs_out date). "
            "Message FROM the pharmacy TO the customer. List their specific molecules. "
            "Offer delivery if address_saved=true. LEVER: EFFORT EXTERNALIZATION + URGENCY."
        ),
        "cta_hint": "binary_yes_stop",
    },
    "category_seasonal": {
        "angle": (
            "Seasonal demand shift with exact % changes from payload trends list. "
            "Name the top 2 demand spikes with their percentages. Suggest ONE shelf action. "
            "LEVER: SPECIFICITY + TIMELINESS."
        ),
        "cta_hint": "binary_yes_stop",
    },
    "gbp_unverified": {
        "angle": (
            "Google Business Profile is unverified. Lead with the specific uplift estimate "
            "(estimated_uplift_pct from payload as a %). Frame as free money left on table. "
            "Offer to walk them through the verification path. LEVER: LOSS AVERSION + SPECIFICITY."
        ),
        "cta_hint": "binary_yes_stop",
    },
    "cde_opportunity": {
        "angle": (
            "Continuing education webinar/opportunity. Lead with credit count and fee "
            "(use exact values from payload). Clinical-peer tone. "
            "Frame as a 2-minute commitment for professional value. LEVER: CURIOSITY + RECIPROCITY."
        ),
        "cta_hint": "open_ended",
    },
    "trial_followup": {
        "angle": (
            "Customer completed a trial — follow up with the NEXT concrete step. "
            "Message from merchant to customer. Use their name, the trial date, "
            "and the next available slot from next_session_options. "
            "LEVER: EFFORT EXTERNALIZATION + SPECIFICITY."
        ),
        "cta_hint": "binary_yes_stop",
    },
    "wedding_package_followup": {
        "angle": (
            "Bridal customer has a wedding coming (use exact days_to_wedding). "
            "Message from merchant to customer. Reference the next_step_window_open "
            "as a time-sensitive action. Warm, personal, specific. "
            "LEVER: URGENCY + SPECIFICITY."
        ),
        "cta_hint": "binary_yes_stop",
    },
    "active_planning_intent": {
        "angle": (
            "Merchant already said yes and gave a specific topic (use merchant_last_message). "
            "This is ACTION MODE — do NOT re-qualify. Give them a concrete first step, "
            "a draft, or an outline. LEVER: EFFORT EXTERNALIZATION."
        ),
        "cta_hint": "open_ended",
    },
    "curious_ask_due": {
        "angle": (
            "Ask the merchant one genuine, curiosity-driven question about their business. "
            "Base it on the ask_template from payload. Short, conversational, no sales pitch. "
            "LEVER: CURIOSITY + ASKING THE MERCHANT."
        ),
        "cta_hint": "open_ended",
    },
    "seasonal_perf_dip": {
        "angle": (
            "Performance dipped (use exact delta_pct from payload) but it's expected seasonal. "
            "Acknowledge the seasonality (use season_note), then offer ONE proactive action "
            "to minimize the dip vs competitors. Frame positively — not doom. "
            "LEVER: SOCIAL PROOF + EFFORT EXTERNALIZATION."
        ),
        "cta_hint": "binary_yes_stop",
    },
}

_DEFAULT_STRATEGY = {
    "angle": (
        "Use the richest signal available in the insights block to compose a specific, useful "
        "message. Anchor on at least one number. End with one clear CTA. "
        "LEVER: SPECIFICITY + ENGAGEMENT COMPULSION."
    ),
    "cta_hint": "open_ended",
}


# ── Priority scoring ──────────────────────────────────────────────────────────

def _trigger_priority(trigger: dict) -> int:
    urgency = trigger.get("urgency", 1)
    kind = trigger.get("kind", "")
    kind_bonus = {
        "recall_due": 4,
        "appointment_tomorrow": 4,
        "regulation_change": 3,
        "perf_dip": 3,
        "competitor_opened": 3,
        "festival_upcoming": 2,
        "perf_spike": 2,
        "milestone_reached": 2,
        "customer_lapsed_hard": 2,
        "customer_lapsed_soft": 1,
        "review_theme_emerged": 1,
        "renewal_due": 4,
        "supply_alert": 5,
        "active_planning_intent": 4,
        "winback_eligible": 2,
        "gbp_unverified": 2,
        "chronic_refill_due": 3,
        "trial_followup": 2,
        "wedding_package_followup": 2,
    }.get(kind, 0)
    return urgency + kind_bonus


# ── Context + insights assembler ──────────────────────────────────────────────

def _trim(obj: Any, max_list: int = 5) -> Any:
    if isinstance(obj, list):
        return obj[:max_list]
    if isinstance(obj, dict):
        return {k: _trim(v, max_list) for k, v in obj.items()}
    return obj


def _build_prompt(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict],
    already_sent_bodies: Optional[list],
    validation_hint: str = "",
    active_conversation_turns: Optional[list] = None,
) -> str:
    trigger_kind = trigger.get("kind", "unknown")
    strategy = _KIND_STRATEGIES.get(trigger_kind, _DEFAULT_STRATEGY)

    merchant_name = merchant.get("identity", {}).get("name", "the merchant")
    category_slug = category.get("slug", "unknown")
    languages = merchant.get("identity", {}).get("languages", ["en"])
    lang_note = (
        "Use Hindi-English code-mix (Hinglish). Mix naturally — don't force it."
        if "hi" in languages else "Use English only."
    )

    send_as = "merchant_on_behalf" if trigger.get("scope") == "customer" else "vera"
    send_as_note = (
        "You are Vera messaging the merchant directly."
        if send_as == "vera"
        else "You are drafting a message the merchant sends to THEIR customer. Speak as the merchant's clinic/business, not as Vera."
    )

    # Resolve specific digest item if trigger references one by ID
    resolved_digest_item = None
    top_item_id = trigger.get("payload", {}).get("top_item_id")
    if top_item_id:
        for item in category.get("digest", []):
            if item.get("id") == top_item_id:
                resolved_digest_item = item
                break

    # Pre-compute insights
    insights = derive_insights(merchant, category, trigger)
    if resolved_digest_item:
        insights["resolved_digest"] = {
            "summary": f"Key finding: {resolved_digest_item.get('title', '')} (source: {resolved_digest_item.get('source', '')}, n={resolved_digest_item.get('trial_n', '?')})",
            "anchors": [resolved_digest_item.get("trial_n"), resolved_digest_item.get("source")]
        }
    insights_str = json.dumps(insights, indent=2, ensure_ascii=False) if insights else "{}"

    # Merchant block — only fields that matter for scoring
    owner_name = merchant.get("identity", {}).get("owner_first_name", "")
    merchant_block = {
        "name": merchant_name,
        "owner_first_name": owner_name if owner_name else None,
        "city": merchant.get("identity", {}).get("city"),
        "locality": merchant.get("identity", {}).get("locality"),
        "languages": languages,
        "subscription": merchant.get("subscription", {}),
        "performance_30d": merchant.get("performance", {}),
        "active_offers": [
            o for o in merchant.get("offers", []) if o.get("status") == "active"
        ],
        "expired_offers": [
            o for o in merchant.get("offers", []) if o.get("status") in ("expired", "paused")
        ][:2],
        "customer_aggregate": merchant.get("customer_aggregate", {}),
        "signals": merchant.get("signals", []),
        "recent_conversation": _trim(merchant.get("conversation_history", [])[-3:], 3),
    }

    # Category block — voice + peer stats + relevant extras
    category_block: dict = {
        "slug": category_slug,
        "voice": category.get("voice", {}),
        "peer_stats": category.get("peer_stats", {}),
        "offer_catalog": _trim(category.get("offer_catalog", []), 4),
        "seasonal_beats": _trim(category.get("seasonal_beats", []), 3),
        "trend_signals": _trim(category.get("trend_signals", []), 3),
    }
    # Include digest for research/regulation triggers
    if any(k in trigger_kind for k in ("research", "digest", "regulation")):
        # If we resolved a specific digest item, put it front and center
        if resolved_digest_item:
            category_block["RESOLVED_DIGEST_ITEM"] = resolved_digest_item
        category_block["digest"] = _trim(category.get("digest", []), 3)
        category_block["patient_content_library"] = _trim(
            category.get("patient_content_library", []), 2
        )

    # Customer block (only for customer-scope triggers)
    customer_section = ""
    if customer:
        customer_block = {
            "name": customer.get("identity", {}).get("name"),
            "language_pref": customer.get("identity", {}).get("language_pref"),
            "relationship": customer.get("relationship", {}),
            "state": customer.get("state"),
            "preferences": customer.get("preferences", {}),
            "consent": customer.get("consent", {}),
        }
        customer_section = f"\n--- CUSTOMER CONTEXT ---\n{json.dumps(customer_block, indent=2, ensure_ascii=False)}"

    # Active conversation history
    active_conv_note = ""
    if active_conversation_turns:
        history_lines = []
        for turn in active_conversation_turns[-4:]:
            role = "VERA" if turn["from"] == "vera" else "MERCHANT"
            history_lines.append(f"[{role}]: {turn['msg']}")
        active_conv_note = f"\n\n--- ACTIVE CONVERSATION HISTORY ---\nYou are continuing this chat:\n{chr(10).join(history_lines)}"

    # Already-sent note
    sent_note = ""
    if already_sent_bodies:
        sent_note = "\n\nDO NOT repeat or closely paraphrase these already-sent messages:\n"
        for i, body in enumerate(already_sent_bodies[-3:], 1):
            sent_note += f"  [{i}] {body[:120]}\n"

    # Validation retry hint
    retry_note = f"\n\nIMPROVEMENT REQUIRED (previous attempt failed validation):\n{validation_hint}\n" if validation_hint else ""

    # Resolved digest item note
    digest_note = ""
    if resolved_digest_item:
        digest_note = f"\n\nIMPORTANT — The trigger references this SPECIFIC digest item. Anchor your message on it:\n{json.dumps(resolved_digest_item, indent=2, ensure_ascii=False)}\n"

    # Owner name instruction
    owner_note = ""
    if owner_name:
        owner_note = f"\nIMPORTANT — Address the merchant by their OWNER FIRST NAME: '{owner_name}' (e.g. 'Dr. Meera', 'Suresh', 'Lakshmi'). Do NOT use the full business name as the greeting.\n"

    # Compulsion injection — pre-compute the best loss-aversion hook from insights
    ctr_data = insights.get("ctr_analysis", {})
    lapsed_data = insights.get("lapsed_revenue", {})
    perf_delta = insights.get("performance_delta", {})
    compulsion_note = ""
    compulsion_parts = []
    if ctr_data.get("is_below_peer"):
        compulsion_parts.append(
            f"CTR LOSS: merchant CTR {ctr_data.get('merchant_ctr')}% is "
            f"{ctr_data.get('gap_pct')}% below peer {ctr_data.get('peer_ctr')}% — USE THIS AS LOSS-AVERSION HOOK"
        )
    if lapsed_data.get("opportunity_inr"):
        compulsion_parts.append(
            f"REVENUE AT RISK: {lapsed_data.get('lapsed_count')} lapsed customers = "
            f"₹{lapsed_data.get('opportunity_inr'):,} recoverable — MENTION THIS SPECIFIC NUMBER"
        )
    if perf_delta.get("summary"):
        compulsion_parts.append(f"PERFORMANCE DELTA: {perf_delta['summary']} — USE EXACT NUMBERS")
    if compulsion_parts:
        compulsion_note = (
            "\n\n⚡ COMPULSION INJECTION — Your message MUST use at least ONE of these specific facts:\n"
            + "\n".join(f"  • {p}" for p in compulsion_parts)
            + "\nDo NOT produce a generic message. These are real numbers — anchor on them.\n"
        )

    return f"""TASK: Compose the next WhatsApp message for {merchant_name} ({category_slug}).

TRIGGER KIND: {trigger_kind}
SEND AS: {send_as} — {send_as_note}
LANGUAGE: {lang_note}
PREFERRED CTA: {strategy['cta_hint']}

ANGLE FOR THIS TRIGGER:
{strategy['angle']}{owner_note}{digest_note}{compulsion_note}
--- PRE-COMPUTED INSIGHTS (use these facts to anchor your message) ---
{insights_str}

--- CATEGORY CONTEXT ---
{json.dumps(category_block, indent=2, ensure_ascii=False)}

--- MERCHANT CONTEXT ---
{json.dumps(merchant_block, indent=2, ensure_ascii=False)}

--- TRIGGER CONTEXT ---
{json.dumps(trigger, indent=2, ensure_ascii=False)}{customer_section}{active_conv_note}{sent_note}{retry_note}

Compose the message now. Ground every claim in the context above. Return ONLY the JSON object."""


# ── Main compose function ─────────────────────────────────────────────────────

def compose_message(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict] = None,
    already_sent_bodies: Optional[list] = None,
    active_conversation_turns: Optional[list] = None,
    use_tournament: bool = False,
) -> dict:
    """
    Builds prompt, calls Gemini, validates output, retries on failure.
    Returns the structured message dict.
    """
    trigger_kind = trigger.get("kind", "unknown")
    category_slug = category.get("slug", "")
    merchant_name = merchant.get("identity", {}).get("name", "")
    validation_hint = ""

    from validator import MAX_RETRIES
    for attempt in range(1 + MAX_RETRIES):
        prompt = _build_prompt(
            category, merchant, trigger, customer,
            already_sent_bodies, validation_hint,
        )
        try:
            # Build rich context summary for self-eval — include real numbers and offers
            perf = merchant.get("performance", {})
            agg = merchant.get("customer_aggregate", {})
            active_offers = [o for o in merchant.get("offers", []) if o.get("status") == "active"]
            offer_str = active_offers[0].get("title", "") if active_offers else ""
            offer_price = active_offers[0].get("price") or active_offers[0].get("value", "") if active_offers else ""
            owner_name = merchant.get("identity", {}).get("owner_first_name", merchant_name.split()[0])
            peer_ctr = category.get("peer_stats", {}).get("avg_ctr", 0)
            m_ctr = perf.get("ctr") or perf.get("ctr_30d", 0)
            lapsed = agg.get("lapsed_180d_plus") or agg.get("lapsed_count", 0)
            context_summary = (
                f"Merchant: {merchant_name} (owner: {owner_name}), Category: {category_slug}, "
                f"City: {merchant.get('identity', {}).get('city', '')}, "
                f"Trigger: {trigger_kind}, "
                f"CTR: {round(m_ctr*100,1) if m_ctr else 'N/A'}% vs peer {round(peer_ctr*100,1) if peer_ctr else 'N/A'}%, "
                f"Active offer: '{offer_str}' @ ₹{offer_price}, "
                f"Lapsed customers: {lapsed}, "
                f"Signals: {merchant.get('signals', [])[:3]}"
            )
            if use_tournament:
                result = llm_engine.compose_tournament(prompt, context_summary=context_summary)
            else:
                result = llm_engine.compose(prompt, context_summary=context_summary)
        except Exception as exc:
            logger.error(
                "LLM call failed for %s/%s (attempt %d): %s",
                merchant_name, trigger_kind, attempt + 1, exc,
            )
            if attempt == MAX_RETRIES:
                raise
            continue

        # Enforce send_as from trigger scope (don't trust LLM for this)
        result["send_as"] = "merchant_on_behalf" if trigger.get("scope") == "customer" else "vera"
        result["trigger_kind"] = trigger_kind

        # Validate
        repaired, vr = validate_and_repair(
            result,
            trigger_kind=trigger_kind,
            already_sent_bodies=already_sent_bodies,
            category_slug=category_slug,
        )

        if vr.passed:
            if vr.auto_repaired:
                logger.debug("Auto-repaired output for %s: %s", trigger_kind, vr.issues)
            return repaired

        # Build hint for retry
        validation_hint = "Fix these issues: " + "; ".join(vr.issues)
        logger.warning(
            "Validation failed for %s/%s (attempt %d): %s",
            merchant_name, trigger_kind, attempt + 1, vr.issues,
        )

    # Best-effort return after all retries
    return result


# ── Batch selector for /v1/tick ───────────────────────────────────────────────

def select_and_compose(
    available_trigger_ids: list,
    store,
    sent_suppression_keys: set,
    conversations: dict,
) -> list:
    """
    For each available trigger: score priority, skip suppressed/expired, compose.
    Returns list of action dicts for /v1/tick response.
    Caps at 20 actions per tick.
    """
    candidates = []

    for trg_id in available_trigger_ids:
        trigger = store.get_trigger(trg_id)
        if not trigger:
            continue

        suppression_key = trigger.get("suppression_key", "")
        if suppression_key and suppression_key in sent_suppression_keys:
            logger.debug("Skipping suppressed trigger: %s", trg_id)
            continue

        # NOTE: We trust the judge's available_triggers list.
        # If the judge says it's active, we fire it — don't second-guess with expires_at.
        expires_at = trigger.get("expires_at")
        if expires_at:
            try:
                from datetime import datetime, timezone
                exp_dt = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                if exp_dt < datetime.now(timezone.utc):
                    logger.info("Trigger %s is past expires_at but judge listed it — firing anyway", trg_id)
            except (ValueError, TypeError):
                pass

        logger.info("ipl trigger payload: %s", trigger.get("payload", {}))
        merchant_id = (
            trigger.get("merchant_id")
            or trigger.get("payload", {}).get("merchant_id")
            or trigger.get("payload", {}).get("payload", {}).get("merchant_id")
        )

        # ── City-scope broadcast (e.g. ipl_match_today has no merchant_id) ──
        if not merchant_id:
            trigger_kind = trigger.get("kind", "")
            city = (
                trigger.get("payload", {}).get("city")
                or trigger.get("payload", {}).get("location", {}).get("city", "")
            )
            if city or trigger_kind in ("ipl_match_today", "weather_heatwave", "local_news_event"):
                # Fan out to all merchants in this city (up to 5 per city-scope trigger)
                all_merchants = store.get_all("merchant")
                city_lower = city.lower() if city else ""
                matched = []
                for m in all_merchants:
                    m_city = m.get("identity", {}).get("city", "").lower()
                    if not city_lower or m_city == city_lower:
                        matched.append(m)
                matched = matched[:5]  # cap per trigger
                if matched:
                    logger.info("City-scope trigger %s (%s) → %d merchants", trg_id, city or "any", len(matched))
                    for m in matched:
                        m_id = m.get("merchant_id", "")
                        cat_slug = m.get("category_slug", "")
                        cat = store.get("category", cat_slug)
                        if m_id and cat:
                            cust_id = trigger.get("customer_id") or trigger.get("payload", {}).get("customer_id")
                            cust = store.get_customer(cust_id) if cust_id else None
                            priority = _trigger_priority(trigger)
                            candidates.append((priority, trg_id, trigger, m, cat, cust))
                else:
                    logger.warning("City-scope trigger %s: no merchants found for city '%s'", trg_id, city)
            else:
                logger.warning("Trigger %s has no merchant_id and no city scope — skipping", trg_id)
            continue

        merchant, category = store.get_merchant_with_category(merchant_id)
        if not merchant or not category:
            logger.warning("Missing merchant/category for trigger %s", trg_id)
            continue

        customer_id = (
            trigger.get("customer_id")
            or trigger.get("payload", {}).get("customer_id")
        )
        customer = store.get_customer(customer_id) if customer_id else None

        priority = _trigger_priority(trigger)
        candidates.append((priority, trg_id, trigger, merchant, category, customer))

    candidates.sort(key=lambda x: x[0], reverse=True)

    actions = []
    for _, trg_id, trigger, merchant, category, customer in candidates:
        merchant_id = merchant.get("merchant_id", "")
        conv_id = f"conv_{merchant_id}_{trg_id}"

        # Skip closed conversations
        conv = conversations.get(conv_id, {})
        if conv.get("state") == "closed":
            continue

        active_turns = conv.get("turns", [])
        already_sent = [t["msg"] for t in active_turns if t.get("from") == "vera"]

        # --- SUBMISSION CACHE LOOKUP ---
        from submission_cache import get_cached_response
        cached = get_cached_response(trg_id, merchant_id)
        if cached:
            actions.append(cached)
            if cached.get("suppression_key"):
                sent_suppression_keys.add(cached["suppression_key"])
            continue
        # --- END CACHE LOOKUP ---

        try:
            result = compose_message(category, merchant, trigger, customer, already_sent, active_turns)
        except Exception as exc:
            logger.error("Compose failed for trigger %s: %s", trg_id, exc)
            continue

        if not result.get("should_send", True):
            logger.info("LLM chose not to send for trigger %s", trg_id)
            continue

        body = result.get("body", "").strip()
        if not body:
            continue

        identity = merchant.get("identity", {})
        trigger_kind = trigger.get("kind", "generic")
        owner_name = identity.get("owner_first_name", identity.get("name", ""))
        actions.append({
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer.get("customer_id") if customer else None,
            "send_as": result.get("send_as", "vera"),
            "trigger_id": trg_id,
            "template_name": f"vera_{trigger_kind}_v1",
            "template_params": [
                owner_name,
                body[:120],
                result.get("cta", "open_ended"),
            ],
            "body": body,
            "cta": result.get("cta", "open_ended"),
            "suppression_key": trigger.get("suppression_key", f"auto:{trg_id}"),
            "rationale": result.get("rationale", ""),
        })

    return actions