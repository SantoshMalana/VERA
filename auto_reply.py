"""
auto_reply_v2.py — Upgraded auto-reply detection
====================================================
Key improvements over v1:
  1. Fuzzy similarity check catches paraphrased auto-replies (not just exact verbatim)
  2. Structural heuristics: length buckets, formal-close phrases, impersonal language
  3. Confidence scoring (not binary) — LOW/MEDIUM/HIGH auto-reply confidence
  4. Tracks if message is IDENTICAL to a VERA message (mistaken routing detection)
  5. Removes the 3-strike counter reliance — better to classify correctly on turn 1

Drop this file in your vera-bot directory as auto_reply.py (replaces old one)
"""
from __future__ import annotations
import re
from difflib import SequenceMatcher

# ── Hard-match patterns (high confidence) ────────────────────────────────────

_HIGH_CONF_PATTERNS = [
    r"thank you for contacting",
    r"thanks for (contacting|reaching out|getting in touch)",
    r"we have received your (message|enquiry|query|request)",
    r"our team will (get back|respond|reply|contact)",
    r"will get back to you (shortly|soon|within)",
    r"currently (unavailable|away|out of office|busy)",
    r"this is an automated (message|reply|response)",
    r"main ek automated (assistant|bot)",
    r"automated assistant hoon",
    r"aapki jaankari ke liye bahut.{0,10}shukriya.*team tak pahuncha",
    r"main aapki.*baatein.*team tak pahuncha",
    r"hamari team.*se sampark karegi",
    r"please (leave|send) (us )?your (details|name|number)",
    r"do not reply to this (message|number)",
    r"this number is not monitored",
    r"i am an automated assistant",
    r"ye ek swachalit sandesh hai",
    r"auto.?reply",
    r"bot se baat kar rahe hain",
]

# ── Soft-match patterns (medium confidence, need corroboration) ───────────────

_MEDIUM_CONF_PATTERNS = [
    r"business hours",
    r"office hours",
    r"working hours",
    r"kaam ke ghante",
    r"hum aapse jald sampark",
    r"bahut bahut shukriya",
    r"aapka sandesh prapt",
    r"response ke liye dhanyawad",
    r"we appreciate your (message|patience|interest)",
    r"your query has been (registered|received|noted)",
]

# ── Impersonal/formal close phrases ──────────────────────────────────────────

_FORMAL_CLOSE_PATTERNS = [
    r"team tak pahuncha (denge|deti|deta)",
    r"team ko inform",
    r"forward kar (denge|deti|deta)",
    r"management ko bata",
    r"sahyogi se baat",
    r"concerned person",
]

_HIGH_COMPILED = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _HIGH_CONF_PATTERNS]
_MED_COMPILED = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _MEDIUM_CONF_PATTERNS]
_FORMAL_COMPILED = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _FORMAL_CLOSE_PATTERNS]


def _fuzzy_similarity(a: str, b: str) -> float:
    """Returns 0.0-1.0 string similarity ratio."""
    a_norm = re.sub(r"\s+", " ", a.lower().strip())
    b_norm = re.sub(r"\s+", " ", b.lower().strip())
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def _structural_score(message: str) -> float:
    """
    Returns 0.0-1.0 structural auto-reply likelihood.
    Based on: length, formality markers, impersonal language.
    """
    score = 0.0
    msg = message.strip()
    length = len(msg)

    # Auto-replies tend to be 40-200 chars (not too short, not too long)
    if 40 <= length <= 250:
        score += 0.15

    # Lacks question marks (auto-replies never ask questions)
    if "?" not in msg:
        score += 0.1

    # Lacks personal pronouns (merchant would say "I", "mujhe", "main")
    personal_pronouns = re.compile(r"\b(i will|i can|main|mujhe|meri|mera|hamara|hum)\b", re.IGNORECASE)
    if not personal_pronouns.search(msg):
        score += 0.1

    # Formal close phrases
    if any(p.search(msg) for p in _FORMAL_COMPILED):
        score += 0.25

    return min(score, 1.0)


def auto_reply_confidence(message: str) -> tuple[str, float]:
    """
    Returns (label, confidence) where label is 'auto' or 'real'
    and confidence is 0.0-1.0.
    """
    if not message or not message.strip():
        return "real", 0.0

    msg = message.strip()

    # Hard-match: high confidence auto-reply
    if any(p.search(msg) for p in _HIGH_COMPILED):
        return "auto", 0.95

    # Medium match
    med_count = sum(1 for p in _MED_COMPILED if p.search(msg))
    structural = _structural_score(msg)

    combined = (med_count * 0.2) + structural
    if combined >= 0.4:
        return "auto", min(combined, 0.9)

    return "real", 1.0 - combined


def is_auto_reply(message: str) -> bool:
    """Drop-in replacement for v1 is_auto_reply. Returns True if likely auto-reply."""
    label, conf = auto_reply_confidence(message)
    return label == "auto" and conf >= 0.7


def is_repeated_message(history: list[dict], message: str, threshold: int = 2) -> bool:
    """
    Return True if:
    - exact same message appeared >= threshold times from merchant side, OR
    - fuzzy-similar (>= 0.85) message appeared >= threshold times
    """
    if not message or not message.strip():
        return False

    normalized = message.strip().lower()
    merchant_turns = [
        t.get("msg", "")
        for t in history
        if t.get("from") in ("merchant", "customer")
    ]

    # Exact match count
    exact_count = sum(1 for m in merchant_turns if m.strip().lower() == normalized)
    if exact_count >= threshold:
        return True

    # Fuzzy match count (catches slight paraphrases)
    fuzzy_count = sum(
        1 for m in merchant_turns
        if _fuzzy_similarity(m, message) >= 0.85
    )
    return fuzzy_count >= threshold


def is_vera_message_echo(message: str, vera_turns: list[str]) -> bool:
    """
    Detect if the merchant is echoing back one of Vera's own messages.
    This can happen with certain WhatsApp bots that re-send received messages.
    """
    for vera_msg in vera_turns:
        if _fuzzy_similarity(message, vera_msg) >= 0.80:
            return True
    return False


def classify_reply(message: str, history: list[dict]) -> str:
    """
    Classify a merchant/customer reply.
    Returns: 'auto_reply' | 'repeated_auto_reply' | 'real'

    Upgrade from v1: uses confidence scoring + fuzzy repeat detection.
    """
    # Check if it echoes a known Vera message (misrouted echo)
    vera_turns = [t.get("msg", "") for t in history if t.get("from") == "vera"]
    if is_vera_message_echo(message, vera_turns):
        return "auto_reply"

    # Auto-reply pattern check
    label, conf = auto_reply_confidence(message)
    if label == "auto" and conf >= 0.7:
        # High confidence: immediate
        if conf >= 0.85:
            return "auto_reply"
        # Medium confidence: also check if it's repeated
        if is_repeated_message(history, message, threshold=1):
            return "repeated_auto_reply"
        return "auto_reply"

    # Real-but-repeated check
    if is_repeated_message(history, message, threshold=2):
        return "repeated_auto_reply"

    return "real"
