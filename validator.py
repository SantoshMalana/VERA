"""
Post-LLM output validation and repair — v2.

Adds:
  - Compulsion Pre-Filter (Fix 4)
  - Category Voice Enforcement (Fix 5) with surgical repair
"""
from __future__ import annotations
import re
from typing import Optional, Callable

MAX_RETRIES = 2

# ── Specificity ───────────────────────────────────────────────────────────────
_SPECIFICITY_PATTERNS = [r"\d+", r"₹\s*\d+", r"\d+\s*%", r"\d+\s*star", r"\d+\s*review"]
_SPECIFICITY_COMPILED = [re.compile(p, re.IGNORECASE) for p in _SPECIFICITY_PATTERNS]

def _has_specificity(body: str) -> bool:
    return any(p.search(body) for p in _SPECIFICITY_COMPILED)

def _number_in_first_sentence(body: str) -> bool:
    first = re.split(r"[.!?।]", body.strip(), maxsplit=1)[0]
    return bool(re.search(r"\d", first))

# ── Hallucination ─────────────────────────────────────────────────────────────
_HALLUCINATION_COMPILED = [re.compile(p, re.IGNORECASE) for p in [
    r"\bI checked (online|google|the web)\b", r"\bI found that\b",
    r"\baccording to my (research|search)\b", r"\bI searched\b", r"\bguaranteed\b",
]]

def _has_hallucination_marker(body: str) -> bool:
    return any(p.search(body) for p in _HALLUCINATION_COMPILED)

# ── URL ───────────────────────────────────────────────────────────────────────
_URL_PATTERN = re.compile(
    r'https?://[^\s]+|www\.[^\s]+|[a-zA-Z0-9-]+\.(com|in|org|net|io|app|co)/[^\s]*'
    r'|\[link[^\]]*\]|\[url[^\]]*\]|\[click here[^\]]*\]|\[draft[^\]]*\]|\[post[^\]]*\]',
    re.IGNORECASE,
)

def _has_url(body: str) -> bool:
    return bool(_URL_PATTERN.search(body))

def _strip_urls(body: str) -> str:
    return _URL_PATTERN.sub('', body).strip()

# ── CTA shape ─────────────────────────────────────────────────────────────────
_ACTION_TRIGGER_KINDS = {
    "perf_spike", "perf_dip", "festival_upcoming", "competitor_opened",
    "recall_due", "customer_lapsed_soft", "customer_lapsed_hard",
    "regulation_change", "category_trend_movement", "review_theme_emerged",
    "local_news_event", "ipl_match_today",
}

def _validate_cta(cta: str, trigger_kind: str) -> tuple[bool, str]:
    if trigger_kind in _ACTION_TRIGGER_KINDS:
        if cta not in ("binary_yes_stop", "open_ended"):
            return False, "binary_yes_stop"
    if trigger_kind == "appointment_tomorrow":
        if cta != "none":
            return False, "none"
    return True, cta

# ── Preamble ──────────────────────────────────────────────────────────────────
_PREAMBLE_COMPILED = [re.compile(p, re.IGNORECASE) for p in [
    r"^I hope (you'?re|you are) (doing well|having a good)",
    r"^Hope (this|the day) finds you",
    r"^Dear (merchant|partner|doctor|owner)",
    r"^Greetings!", r"^Hello! I am Vera", r"^Hi! I'?m? Vera",
    r"^I wanted to reach out", r"^Just checking in",
]]

def _has_preamble(body: str) -> bool:
    return any(p.search(body.strip()) for p in _PREAMBLE_COMPILED)

# ── Length ────────────────────────────────────────────────────────────────────
MIN_BODY_LENGTH = 30
MAX_BODY_LENGTH = 800
_HIGH_URGENCY_KINDS = {"perf_dip", "competitor_opened", "ipl_match_today", "festival_upcoming"}
HIGH_URGENCY_SOFT_CAP = 320

# ── Compulsion signals ────────────────────────────────────────────────────────
_LOSS_AVERSION_COMPILED = [re.compile(p, re.IGNORECASE) for p in [
    r"\bmiss(ing|ed)?\b", r"\blose\b", r"\blost\b", r"\blosing\b", r"\bgap\b",
    r"\bbelow (peer|average|median)\b", r"\bdrop(ped|ping)?\b", r"\bfalling\b",
    r"\bpeeche\b", r"\bgir\b", r"\bkam\b",
    r"\bopportunity\b", r"\bchance\b",
    r"\bbefore .{0,20}(close|end|expire)\b",
    r"\bsirf .{0,10}(din|days?|ghante)\b",
]]

_WARM_CLOSE_COMPILED = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in [
    r"\?[\s\S]{0,10}$",          # ? anywhere near the end (allows trailing emoji/space)
    r"\bYES\b",                   # explicit YES CTA
    r"\bSTOP\b",                  # explicit STOP CTA
    r"\breply (1|2|yes|no)\b",   # slot-pick reply
    r"\bkya kehte\b",             # Hinglish engagement hook
    r"\bchahiye\b",               # "do you want?" in Hindi
    r"\bkarein\b",                # "shall we do?" in Hindi
    r"\bboliye\b",                # "please tell" in Hindi
    r"\bkar doon\b",              # "shall I do?" in Hindi
    r"\bbhej doon\b",             # "shall I send?" in Hindi
    r"\bblock kar doon\b",        # "shall I block?" in Hindi
    r"\bdraft kar doon\b",        # "shall I draft?" in Hindi
]]

def _has_loss_aversion(body: str) -> bool:
    return any(p.search(body) for p in _LOSS_AVERSION_COMPILED)

def _has_warm_close(body: str) -> bool:
    return any(p.search(body) for p in _WARM_CLOSE_COMPILED)

# ── Category Voice Enforcement ────────────────────────────────────────────────
_SLUG_NORMALIZE = {
    "dentist": "dentists", "salon": "salons", "gym": "gyms",
    "restaurant": "restaurants", "pharmacy": "pharmacies",
}

_CATEGORY_FORBIDDEN: dict[str, list[tuple[re.Pattern, str]]] = {
    "dentists": [
        (re.compile(r"\bcure\b", re.IGNORECASE), "treat"),
        (re.compile(r"\bguaranteed\b", re.IGNORECASE), "evidence-based"),
        (re.compile(r"\b100% safe\b", re.IGNORECASE), "clinically reviewed"),
        (re.compile(r"\bpainless guaranteed\b", re.IGNORECASE), "minimally invasive"),
        (re.compile(r"\bboost(ing)?\b", re.IGNORECASE), "improve"),
        (re.compile(r"\bskyrocket\b", re.IGNORECASE), "increase"),
        (re.compile(r"\bamazing results\b", re.IGNORECASE), "strong outcomes"),
        (re.compile(r"\bincredible\b", re.IGNORECASE), "notable"),
    ],
    "pharmacies": [
        (re.compile(r"\bcure\b", re.IGNORECASE), "treat"),
        (re.compile(r"\bguaranteed\b", re.IGNORECASE), "clinically indicated"),
        (re.compile(r"\b100% safe\b", re.IGNORECASE), "FDA approved"),
        (re.compile(r"\bmiracl\w*\b", re.IGNORECASE), "effective"),
    ],
    "salons": [
        (re.compile(r"\bclinical trial\b", re.IGNORECASE), "professional treatment"),
        (re.compile(r"\bpatient\b", re.IGNORECASE), "client"),
    ],
    "gyms": [
        (re.compile(r"\bpatient\b", re.IGNORECASE), "member"),
        (re.compile(r"\bclinical\b", re.IGNORECASE), "science-backed"),
    ],
    "restaurants": [
        (re.compile(r"\bpatient\b", re.IGNORECASE), "guest"),
        (re.compile(r"\bclinical\b", re.IGNORECASE), "curated"),
    ],
}

_EMOJI_PATTERN = re.compile(
    "[\U0001F300-\U0001F9FF\U00002600-\U000027BF\U0001FA00-\U0001FA9F]",
    re.UNICODE,
)
_CATEGORY_EMOJI = {
    "dentists": "🦷", "salons": "💇", "restaurants": "🍽️",
    "gyms": "💪", "pharmacies": "💊",
}


def _enforce_category_voice(body: str, category_slug: str) -> tuple[str, list[str]]:
    """Surgical category voice enforcement. Returns (repaired_body, repairs_list)."""
    repairs = []
    slug = _SLUG_NORMALIZE.get(category_slug, category_slug)

    for pattern, replacement in _CATEGORY_FORBIDDEN.get(slug, []):
        new_body, n = pattern.subn(replacement, body)
        if n > 0:
            repairs.append(f"Category voice: replaced '{pattern.pattern}' → '{replacement}'")
            body = new_body

    # Emoji discipline — one approved emoji at the end only
    approved = _CATEGORY_EMOJI.get(slug)
    if approved:
        found = _EMOJI_PATTERN.findall(body)
        if found:
            body = _EMOJI_PATTERN.sub("", body).strip().rstrip("!.? ")
            body = body + f" {approved}"
            repairs.append(f"Emoji disciplined to single {approved} at end")

    return body, repairs


# ── Compulsion score (advisory) ───────────────────────────────────────────────
def _compulsion_score(body: str, trigger_kind: str, owner_name: str = "") -> dict:
    score: dict = {}
    score["number_in_first_sentence"] = _number_in_first_sentence(body)
    score["loss_aversion"] = _has_loss_aversion(body)
    score["warm_close"] = _has_warm_close(body)
    first_words = " ".join(body.split()[:8])
    score["owner_name_early"] = (
        bool(owner_name and owner_name.split()[0].lower() in first_words.lower())
        if owner_name else True
    )
    cta_signals = re.findall(
        r"\b(YES|STOP|reply (1|2|yes|no)|kya kehte|chahiye)\b", body, re.IGNORECASE
    )
    score["single_cta"] = len(cta_signals) <= 1
    if trigger_kind in _HIGH_URGENCY_KINDS:
        score["urgency_length_ok"] = len(body) <= HIGH_URGENCY_SOFT_CAP
    return score


# ── ValidationResult ──────────────────────────────────────────────────────────
class ValidationResult:
    def __init__(self):
        self.passed = True
        self.issues: list[str] = []
        self.auto_repaired = False
        self.compulsion_scores: dict = {}

    def fail(self, issue: str):
        self.passed = False
        self.issues.append(issue)

    def repair(self, issue: str):
        self.auto_repaired = True
        self.issues.append(f"[auto-repaired] {issue}")

    def advisory(self, issue: str):
        self.issues.append(f"[advisory] {issue}")


# ── Main ──────────────────────────────────────────────────────────────────────
def validate_and_repair(
    result: dict,
    trigger_kind: str,
    already_sent_bodies: Optional[list] = None,
    category_slug: str = "",
    owner_name: str = "",
) -> tuple[dict, ValidationResult]:
    vr = ValidationResult()
    body = result.get("body", "").strip()
    cta = result.get("cta", "open_ended")

    if not body:
        vr.fail("Empty body"); return result, vr

    if len(body) < MIN_BODY_LENGTH:
        vr.fail(f"Body too short ({len(body)} chars)"); return result, vr

    if len(body) > MAX_BODY_LENGTH:
        truncated = body[:MAX_BODY_LENGTH]
        last_end = max(truncated.rfind("."), truncated.rfind("?"), truncated.rfind("!"))
        if last_end > MAX_BODY_LENGTH // 2:
            body = truncated[:last_end + 1]
            result["body"] = body
            vr.repair(f"Truncated to {len(body)} chars")

    if already_sent_bodies:
        for prev in already_sent_bodies:
            if prev.strip().lower() == body.lower():
                vr.fail("Verbatim repeat"); return result, vr

    if not _has_specificity(body):
        vr.fail("No specificity anchor (numbers/prices/%)")

    if _has_hallucination_marker(body):
        vr.fail("Hallucination marker detected"); return result, vr

    if _has_url(body):
        cleaned = _strip_urls(body)
        if cleaned and len(cleaned) >= MIN_BODY_LENGTH:
            result["body"] = cleaned; body = cleaned
            vr.repair("Stripped URLs")
        else:
            vr.fail("Unstrippable URLs"); return result, vr

    if _has_preamble(body):
        sentences = re.split(r"(?<=[.!?])\s+", body, maxsplit=1)
        if len(sentences) > 1:
            body = sentences[1].strip(); result["body"] = body
            vr.repair("Stripped preamble")
        else:
            vr.fail("Unstrippable preamble")

    # Category voice enforcement (surgical)
    if category_slug:
        body, cat_repairs = _enforce_category_voice(body, category_slug)
        result["body"] = body
        for r in cat_repairs:
            vr.repair(r)

    # CTA shape
    cta_valid, corrected_cta = _validate_cta(cta, trigger_kind)
    if not cta_valid:
        result["cta"] = corrected_cta
        vr.repair(f"CTA '{cta}' → '{corrected_cta}'")

    # Multiple CTA signals
    yes_no_count = len(re.findall(
        r"\b(YES|STOP|reply yes|reply no|reply 1|reply 2)\b", body, re.IGNORECASE
    ))
    if yes_no_count > 2:
        vr.fail("Multiple CTA signals")

    # Medical taboo (post-repair check)
    if category_slug in {"dentists", "pharmacies", "doctors"}:
        taboo = [w for w in ["cure", "guaranteed", "100% safe"] if w.lower() in body.lower()]
        if taboo:
            vr.fail(f"Medical taboo after repair: {taboo}")

    # Compulsion advisory
    comp = _compulsion_score(body, trigger_kind, owner_name)
    vr.compulsion_scores = comp
    if not comp.get("number_in_first_sentence"):
        vr.advisory("No number in first sentence")
    _WARM_CLOSE_EXEMPT = {"appointment_tomorrow", "none", "chronic_refill_due", "trial_followup", "wedding_package_followup"}
    if not comp.get("warm_close") and trigger_kind not in _WARM_CLOSE_EXEMPT:
        vr.fail("No warm close — message must end with a question or YES/STOP CTA")
    if trigger_kind in _HIGH_URGENCY_KINDS and not comp.get("urgency_length_ok", True):
        vr.advisory(f"High-urgency body {len(body)} chars > {HIGH_URGENCY_SOFT_CAP} soft cap")

    critical_failures = [i for i in vr.issues
                         if not i.startswith("[auto-repaired]") and not i.startswith("[advisory]")]
    if not critical_failures:
        vr.passed = True

    return result, vr


def validate_with_retry(
    compose_fn: Callable,
    trigger_kind: str,
    already_sent_bodies: Optional[list] = None,
    category_slug: str = "",
) -> dict:
    last_result = None
    for _ in range(1 + MAX_RETRIES):
        result = compose_fn()
        repaired, vr = validate_and_repair(
            result, trigger_kind=trigger_kind,
            already_sent_bodies=already_sent_bodies, category_slug=category_slug,
        )
        if vr.passed:
            return repaired
        last_result = repaired
    return last_result or {}