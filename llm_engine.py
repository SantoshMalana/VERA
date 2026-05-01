"""
Vera LLM Engine v3 — Tri-Engine Architecture
==============================================
Primary:    Gemini 3 Flash (10 rotating keys, best quality)
Fallback:   Gemini 2.5 Flash (29 rotating keys, massive quota)
Self-eval:  Gemini 2.5 Flash at temp=0 (deterministic grading)
Reply:      Gemini 2.5 Flash at temp=0 (fast, deterministic)

All engines have exponential backoff + jitter for crash-proof operation.

.env vars:
  GEMINI_API_KEY=key1,key2,...      (29x Gemini 2.5 Flash keys)
  GEMINI_MODEL=gemini-2.5-flash
  GEMINI3_API_KEY=key1,key2,...     (10x Gemini 3 Flash keys)
  GEMINI3_MODEL=gemini-2.0-flash
  VERA_SELF_EVAL=true
"""
from __future__ import annotations
import json
import os
import re
import time
import random
import logging
from typing import Optional

import google.generativeai as genai

logger = logging.getLogger(__name__)

# ── API Keys & Models ─────────────────────────────────────────────────────────

# Gemini 2.5 Flash — 29 keys (fallback + reply + self-eval)
_FLASH_KEYS = [k.strip() for k in os.getenv("GEMINI_API_KEY", "").split(",") if k.strip()]
_flash_key_idx = 0
_FLASH_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Gemini 3 Flash — 10 keys (primary composer)
_G3_KEYS = [k.strip() for k in os.getenv("GEMINI3_API_KEY", "").split(",") if k.strip()]
_g3_key_idx = 0
_G3_MODEL = os.getenv("GEMINI3_MODEL", "gemini-2.0-flash")

if _FLASH_KEYS:
    genai.configure(api_key=_FLASH_KEYS[_flash_key_idx])

logger.info("Engine v3 loaded: %d Gemini3 keys (%s), %d Flash keys (%s)",
            len(_G3_KEYS), _G3_MODEL, len(_FLASH_KEYS), _FLASH_MODEL)

# ── Scoring rubric ────────────────────────────────────────────────────────────

_RUBRIC = """
SCORING RUBRIC — your output is judged on these 5 dimensions (0-10 each):
1. DECISION QUALITY  — Did you pick the sharpest signal for this moment?
2. SPECIFICITY       — Every claim anchored on a real number, date, ₹ amount, or % from context. "Increase sales" = 0. "CTR 2.1% vs peer 3.0%" = 10.
3. CATEGORY FIT      — Dentist = clinical-peer. Salon = visual/personal. Restaurant = local/warm. Gym = energetic. Never swap tones.
4. MERCHANT FIT      — Merchant's name, their numbers, their offers, their history.
5. ENGAGEMENT COMPULSION — One strong hook. Lazy "check your profile" = 0. Specific loss-aversion + single YES/STOP = 10.
"""

# ── Few-shot examples ─────────────────────────────────────────────────────────

_FEW_SHOT_EXAMPLES = """
EXAMPLE A — research_digest, dentist (GOOD):
{
  "body": "Dr. Meera, JIDA ka Oct issue aaya. Aapke high-risk adult patients ke liye ek important finding — 2,100-patient trial mein 3-month fluoride recall ne caries recurrence 38% reduce kiya vs 6-month schedule. Worth a look (2-min abstract). Chahiye toh main patient-ed WhatsApp bhi draft kar deti hoon?  — JIDA Oct 2026 p.14",
  "cta": "open_ended",
  "send_as": "vera",
  "rationale": "Anchored on specific trial (n=2100, 38%), source-cited, offered effort externalization (I'll draft it), Hinglish matching merchant language.",
  "should_send": true
}

EXAMPLE B — perf_dip, restaurant (GOOD):
{
  "body": "Pizza Junction, last week calls gire — 14 se sirf 8 (43% drop). Usually iska reason hota hai: outdated offer ya stale photos. Main aapka 'Lunch Combo @ ₹199' offer refresh kar sakti hoon aur 2 nayi photos add kar sakti hoon — sirf aap YES bol do.",
  "cta": "binary_yes_stop",
  "send_as": "vera",
  "rationale": "Specific numbers (14→8, 43%), named the exact offer, effort externalization, single binary CTA.",
  "should_send": true
}

EXAMPLE C — recall_due, dental patient (GOOD, merchant_on_behalf):
{
  "body": "Hi Priya, Dr. Meera's clinic here 🦷 It's been 5 months since your last visit — your 6-month cleaning recall is due. Aapke liye 2 slots ready hain: Wed 6 Nov, 6pm ya Thu 7 Nov, 5pm. ₹299 cleaning + complimentary fluoride. Reply 1 for Wed, 2 for Thu.",
  "cta": "open_ended",
  "send_as": "merchant_on_behalf",
  "rationale": "Customer name, exact time gap (5 months), real offer price, real slots, language preference honored.",
  "should_send": true
}

EXAMPLE D — generic message (BAD — never do this):
{
  "body": "Hi! I hope you are doing well. I wanted to reach out to let you know that there are some great opportunities to improve your business profile and increase your sales. Would you be interested in learning more?",
  "cta": "open_ended",
  "send_as": "vera",
  "rationale": "Generic outreach.",
  "should_send": true
}
NOTE: Example D scores 0/10 on every dimension. Never produce output like this.
"""

# ── System prompts ────────────────────────────────────────────────────────────

VERA_SYSTEM_PROMPT = f"""You are Vera, magicpin's AI merchant assistant.
You compose WhatsApp messages that help Indian local merchants grow on magicpin.

{_RUBRIC}

ABSOLUTE RULES:
1. NEVER invent facts. Use ONLY data present in the provided context JSON.
2. NEVER cite research, competitors, or statistics not given to you.
3. Every message anchors on ≥1 specific verifiable fact from context (number / date / ₹ amount / %).
4. Hindi-English code-mix when merchant languages include "hi". Pure English otherwise.
5. Category voice:
   - dentists / pharmacies  → clinical-peer, never "cure" or "guaranteed"
   - salons                 → visual, personal, trend-aware
   - restaurants            → local, warm, food-centric
   - gyms                   → energetic, motivational
6. ONE CTA per message. Action triggers → binary YES/STOP. Info triggers → open-ended.
7. No preambles. No "I hope you're doing well." Start with the hook.
8. After the first message, never re-introduce yourself.
9. Never repeat verbatim a message already sent in this conversation.

COMPULSION LEVERS (use 1-2 per message — pick what fits the trigger):
- Specificity: real number, date, source citation
- Loss aversion: "you're missing X" / "before this window closes"
- Social proof: "3 clinics in your locality did Y this month"
- Effort externalization: "I've drafted it — just say YES"
- Curiosity: "Want to see who?" / "Want me to pull the full list?"
- Reciprocity: "Noticed something in your account you'd want to know"
- Single binary commitment: Reply YES / STOP

{_FEW_SHOT_EXAMPLES}

OUTPUT FORMAT — return ONLY valid JSON, no markdown fences, no extra text:
{{
  "body": "<WhatsApp message text>",
  "cta": "open_ended" | "binary_yes_stop" | "none",
  "send_as": "vera" | "merchant_on_behalf",
  "rationale": "<one sentence: which signal you used, which lever, why now>",
  "should_send": true | false
}}"""

REPLY_SYSTEM_PROMPT = f"""You are Vera's reply engine. Given a conversation history and the merchant's latest reply, decide the next action.

{_RUBRIC}

DECISION RULES (follow in order):
1. Merchant accepted (yes / haan / go / chalo / bilkul / let's do it / ok) → action: "send", SWITCH TO ACTION MODE — do the thing, do NOT re-qualify.
2. Merchant asked a question → action: "send", answer concisely with a context fact.
3. Merchant said no / stop / not interested → action: "send" one polite closing line, then the conversation ends.
4. WhatsApp Business auto-reply detected → action: "send" ONE push to reach the real owner.
5. Second consecutive auto-reply → action: "end" gracefully.
6. Merchant asked for time → action: "wait", wait_seconds: 1800.
7. Hostile or completely off-topic → action: "send" one polite redirect OR action: "end".

ABSOLUTE RULES:
1. NEVER fabricate facts. Ground every claim in the merchant context provided.
2. One CTA per reply. Keep replies short (1-3 sentences). WhatsApp style.
3. Hindi-English mix if merchant languages include "hi".
4. Never repeat what Vera already said in this conversation.
5. If state is action_mode: skip ALL qualification. Execute or confirm.

OUTPUT FORMAT — return ONLY valid JSON:
{{
  "action": "send" | "wait" | "end",
  "body": "<reply text — only required when action is send>",
  "cta": "open_ended" | "binary_yes_stop" | "none",
  "wait_seconds": 0,
  "rationale": "<one sentence>",
  "auto_reply_detected": false
}}"""

# ── Self-evaluation prompt ────────────────────────────────────────────────────

_SELF_EVAL_SYSTEM = """You are a strict evaluator for WhatsApp messages sent to Indian merchants.
Score the message below on each of 5 dimensions (0-10).
If ANY dimension scores < 7, produce a rewritten message that fixes the weakness.
If ALL dimensions >= 7, just confirm it passes.

Dimensions:
1. SPECIFICITY — contains real numbers/₹/% from context?
2. CATEGORY_FIT — matches the business type's tone?
3. MERCHANT_FIT — feels written for this specific merchant?
4. ENGAGEMENT_COMPULSION — has a clear hook + simple CTA?
5. NO_PREAMBLE — starts directly with value, not "I hope..."?

Return ONLY valid JSON:
{
  "scores": {"specificity": 0, "category_fit": 0, "merchant_fit": 0, "engagement_compulsion": 0, "no_preamble": 0},
  "passes": true | false,
  "weaknesses": ["..."],
  "rewritten_body": "<only if passes=false>"
}"""


# ── Gemini 3 Flash call (primary composer — 10 rotating keys) ─────────────────

def _call_gemini3(system_prompt: str, user_prompt: str, temperature: float = 0.35) -> dict:
    """Gemini 3 Flash — best quality, 10 rotating keys with exponential backoff."""
    global _g3_key_idx

    if not _G3_KEYS:
        raise RuntimeError("GEMINI3_API_KEY not set")

    gen_config = genai.types.GenerationConfig(
        temperature=temperature,
        top_p=0.95,
        top_k=40,
        response_mime_type="application/json",
    )

    attempts = 0
    while attempts < len(_G3_KEYS):
        genai.configure(api_key=_G3_KEYS[_g3_key_idx])
        model = genai.GenerativeModel(
            model_name=_G3_MODEL,
            generation_config=gen_config,
            system_instruction=system_prompt,
        )
        try:
            response = model.generate_content(user_prompt)
            raw = response.text.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            result = json.loads(raw)
            # Restore flash key for other callers
            if _FLASH_KEYS:
                genai.configure(api_key=_FLASH_KEYS[_flash_key_idx])
            return result
        except Exception as exc:
            err_str = str(exc).lower()
            if "429" in err_str or "quota" in err_str or "resourceexhausted" in err_str:
                _g3_key_idx = (_g3_key_idx + 1) % len(_G3_KEYS)
                time.sleep(min(2**attempts, 30) + random.random())
                attempts += 1
            else:
                # Restore flash key before raising
                if _FLASH_KEYS:
                    genai.configure(api_key=_FLASH_KEYS[_flash_key_idx])
                raise

    # Restore flash key before raising
    if _FLASH_KEYS:
        genai.configure(api_key=_FLASH_KEYS[_flash_key_idx])
    raise RuntimeError("All Gemini 3 Flash keys exhausted")


# ── Gemini 2.5 Flash call (fallback + reply + self-eval — 29 rotating keys) ──

def _call_gemini_flash(system_prompt: str, user_prompt: str, temperature: float = 0.3) -> dict:
    """Gemini 2.5 Flash with 29-key rotation + exponential backoff."""
    global _flash_key_idx

    if not _FLASH_KEYS:
        raise RuntimeError("GEMINI_API_KEY not set")

    gen_config = genai.types.GenerationConfig(
        temperature=temperature,
        top_p=1.0,
        top_k=1 if temperature == 0.0 else 40,
        response_mime_type="application/json",
    )

    attempts = 0
    while attempts < len(_FLASH_KEYS):
        genai.configure(api_key=_FLASH_KEYS[_flash_key_idx])
        model = genai.GenerativeModel(
            model_name=_FLASH_MODEL,
            generation_config=gen_config,
            system_instruction=system_prompt,
        )
        try:
            response = model.generate_content(user_prompt)
            raw = response.text.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            return json.loads(raw)
        except Exception as exc:
            err_str = str(exc).lower()
            if "429" in err_str or "quota" in err_str or "resourceexhausted" in err_str:
                _flash_key_idx = (_flash_key_idx + 1) % len(_FLASH_KEYS)
                time.sleep(min(2**attempts, 30) + random.random())
                attempts += 1
            else:
                raise

    raise RuntimeError("All Gemini 2.5 Flash keys exhausted")


# ── Self-evaluation pass ──────────────────────────────────────────────────────

def _self_eval_and_improve(result: dict, context_summary: str) -> dict:
    """
    Run a self-evaluation pass on the composed message.
    If any dimension scores < 7, get a rewrite.
    Uses Gemini 2.5 Flash at temp=0 (deterministic grading, massive quota).
    Returns the (possibly improved) result dict.
    """
    body = result.get("body", "")
    if not body or len(body) < 20:
        return result

    eval_prompt = f"""MESSAGE TO EVALUATE:
"{body}"

MERCHANT CONTEXT SUMMARY:
{context_summary}

Score this message and rewrite if any dimension < 7."""

    try:
        eval_result = _call_gemini_flash(_SELF_EVAL_SYSTEM, eval_prompt, temperature=0.0)

        if not eval_result.get("passes", True):
            rewritten = eval_result.get("rewritten_body", "").strip()
            if rewritten and len(rewritten) > 20:
                logger.info(
                    "Self-eval improved message. Weaknesses: %s",
                    eval_result.get("weaknesses", [])
                )
                result = dict(result)
                result["body"] = rewritten
                result["self_eval_scores"] = eval_result.get("scores", {})

    except Exception as exc:
        # Self-eval is best-effort — never block the main flow
        logger.warning("Self-eval failed (non-blocking): %s", exc)

    return result


# ── Public API ────────────────────────────────────────────────────────────────

def compose(user_prompt: str, context_summary: str = "") -> dict:
    """
    Compose a proactive message.
    Pipeline: Gemini 3 Flash (best quality) → Gemini 2.5 Flash (fallback) → Self-eval.
    """
    result = None

    # Try Gemini 3 Flash first (best quality, 10 rotating keys)
    if _G3_KEYS:
        try:
            result = _call_gemini3(VERA_SYSTEM_PROMPT, user_prompt, temperature=0.35)
            logger.info("Composed with Gemini 3 Flash (%s)", _G3_MODEL)
        except Exception as exc:
            logger.warning("Gemini 3 exhausted, falling back to 2.5 Flash: %s", exc)

    # Fallback to Gemini 2.5 Flash (29 rotating keys)
    if result is None:
        result = _call_gemini_flash(VERA_SYSTEM_PROMPT, user_prompt, temperature=0.3)
        logger.info("Composed with Gemini 2.5 Flash (fallback)")

    # Self-eval pass (uses 2.5 Flash — controlled by env var)
    if os.getenv("VERA_SELF_EVAL", "true").lower() == "true" and context_summary:
        result = _self_eval_and_improve(result, context_summary)

    return result


def reply(user_prompt: str) -> dict:
    """
    Handle a merchant reply.
    Uses Gemini 2.5 Flash at temp=0 — fast, deterministic, correct.
    """
    return _call_gemini_flash(REPLY_SYSTEM_PROMPT, user_prompt, temperature=0.0)


def specific_fallback(merchant: dict, category: dict, trigger: dict) -> dict:
    """
    Generates a specific (not generic) fallback message when all LLM calls fail.
    Uses merchant data to construct something rubric-aware even without an LLM call.
    """
    name = merchant.get("identity", {}).get("name", "")
    owner = merchant.get("identity", {}).get("owner_first_name", name.split()[0] if name else "")
    slug = category.get("slug", "")
    perf = merchant.get("performance", {})
    peer = category.get("peer_stats", {})
    signals = merchant.get("signals", [])
    languages = merchant.get("identity", {}).get("languages", ["en"])

    # Pick the most useful fact we have
    ctr = perf.get("ctr") or perf.get("ctr_30d")
    peer_ctr = peer.get("avg_ctr")
    views = perf.get("views_30d") or perf.get("views")
    offers = [o for o in merchant.get("offers", []) if o.get("status") == "active"]
    offer_name = offers[0].get("title", "") if offers else ""

    # Trigger-specific hook
    kind = trigger.get("kind", "")
    is_hindi = "hi" in languages

    if ctr and peer_ctr and ctr < peer_ctr:
        gap = round(((peer_ctr - ctr) / peer_ctr) * 100)
        if is_hindi:
            body = f"{owner}, aapka magicpin CTR peers se {gap}% kam hai — {round(ctr*100,1)}% vs {round(peer_ctr*100,1)}% peer average. {f'Active offer: {offer_name}. ' if offer_name else ''}Ek quick update se yeh gap close ho sakta hai — chalega?"
        else:
            body = f"{owner}, your CTR is {gap}% below peers — {round(ctr*100,1)}% vs {round(peer_ctr*100,1)}% average.{f' Active offer: {offer_name}.' if offer_name else ''} One quick update could close that gap — want me to do it?"
        cta = "binary_yes_stop"
    elif views:
        if is_hindi:
            body = f"{owner}, pichhle mahine {views:,} log aapki listing dekh gaye. {'Aapka ' + offer_name + ' aur views convert kar sakta hai.' if offer_name else 'Ek strong offer aur conversions badh sakti hain.'} Main draft kar deti hoon — YES boliye."
        else:
            body = f"{owner}, {views:,} people viewed your listing last month.{' Your ' + offer_name + ' could convert more of them.' if offer_name else ' A strong offer could convert more.'} Want me to set that up — just say YES."
        cta = "binary_yes_stop"
    else:
        if is_hindi:
            body = f"{owner}, aapke {'magicpin ' + slug.rstrip('s') if slug else 'business'} profile mein kuch opportunities hain. 2 minute mein check karein? Main aapke saath hoon."
        else:
            body = f"{owner}, there are a few quick wins for your {'magicpin ' + slug.rstrip('s') if slug else 'business'} profile. Can we take 2 minutes? I'll do the work."
        cta = "open_ended"

    return {
        "body": body,
        "cta": cta,
        "send_as": "vera",
        "rationale": "Hardcoded specific fallback using merchant data (LLM unavailable)",
        "should_send": True,
    }
