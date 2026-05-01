"""
Post-LLM output validation and repair.

Runs a fast, rule-based quality gate on composed messages BEFORE they are
returned to the judge. If a check fails, it either auto-repairs the output
or triggers a re-prompt (up to MAX_RETRIES times).

Checks:
  1. Body is non-empty and has sufficient length
  2. Body is not verbatim-identical to a previously sent message
  3. Body contains at least one specific anchor (number, price, %)
  4. CTA shape matches trigger kind expectations
  5. No obvious hallucination markers
  6. send_as is consistent with trigger scope
"""
from __future__ import annotations
import re
from typing import Optional, Callable

MAX_RETRIES = 2

# ── Specificity check: body must contain a number, ₹ price, or % ────────────
_SPECIFICITY_PATTERNS = [
    r"\d+",           # any number
    r"₹\s*\d+",       # rupee amount
    r"\d+\s*%",       # percentage
    r"\d+\s*star",    # star rating
    r"\d+\s*review",  # review count
]
_SPECIFICITY_COMPILED = [re.compile(p, re.IGNORECASE) for p in _SPECIFICITY_PATTERNS]


def _has_specificity(body: str) -> bool:
    return any(p.search(body) for p in _SPECIFICITY_COMPILED)


# ── Hallucination marker check ───────────────────────────────────────────────
_HALLUCINATION_MARKERS = [
    r"\bI checked (online|google|the web)\b",
    r"\bI found that\b",
    r"\baccording to my (research|search)\b",
    r"\bI searched\b",
    r"\bguaranteed\b",  # taboo for medical categories
]
_HALLUCINATION_COMPILED = [re.compile(p, re.IGNORECASE) for p in _HALLUCINATION_MARKERS]


def _has_hallucination_marker(body: str) -> bool:
    return any(p.search(body) for p in _HALLUCINATION_COMPILED)


# ── URL detection (hard fail: -3 penalty per URL) ────────────────────────────
_URL_PATTERN = re.compile(
    r'https?://[^\s]+|www\.[^\s]+|[a-zA-Z0-9-]+\.(com|in|org|net|io|app|co)/[^\s]*',
    re.IGNORECASE,
)


def _has_url(body: str) -> bool:
    return bool(_URL_PATTERN.search(body))


def _strip_urls(body: str) -> str:
    """Remove URLs from body text."""
    return _URL_PATTERN.sub('', body).strip()


# ── CTA shape check ───────────────────────────────────────────────────────────
_ACTION_TRIGGER_KINDS = {
    "perf_spike", "perf_dip", "festival_upcoming", "competitor_opened",
    "recall_due", "customer_lapsed_soft", "customer_lapsed_hard",
    "regulation_change", "category_trend_movement", "review_theme_emerged",
    "local_news_event",
}
_INFO_TRIGGER_KINDS = {
    "research_digest", "research_digest_release",
    "category_research_digest_release", "milestone_reached",
    "dormant_with_vera", "scheduled_recurring", "weather_heatwave",
    "appointment_tomorrow",
}


def _validate_cta(cta: str, trigger_kind: str) -> tuple[bool, str]:
    """Returns (is_valid, corrected_cta)."""
    if trigger_kind in _ACTION_TRIGGER_KINDS:
        if cta not in ("binary_yes_stop", "open_ended"):
            return False, "binary_yes_stop"
    if trigger_kind == "appointment_tomorrow":
        if cta != "none":
            return False, "none"
    return True, cta


# ── Preamble check ────────────────────────────────────────────────────────────
_PREAMBLE_PATTERNS = [
    r"^I hope (you'?re|you are) (doing well|having a good)",
    r"^Hope (this|the day) finds you",
    r"^Dear (merchant|partner|doctor|owner)",
    r"^Greetings!",
    r"^Hello! I am Vera",
    r"^Hi! I'?m? Vera",
]
_PREAMBLE_COMPILED = [re.compile(p, re.IGNORECASE) for p in _PREAMBLE_PATTERNS]


def _has_preamble(body: str) -> bool:
    return any(p.search(body.strip()) for p in _PREAMBLE_COMPILED)


# ── Length check ──────────────────────────────────────────────────────────────
MIN_BODY_LENGTH = 30
MAX_BODY_LENGTH = 800


# ── Main validation function ──────────────────────────────────────────────────

class ValidationResult:
    def __init__(self):
        self.passed = True
        self.issues: list[str] = []
        self.auto_repaired = False

    def fail(self, issue: str):
        self.passed = False
        self.issues.append(issue)

    def repair(self, issue: str):
        self.auto_repaired = True
        self.issues.append(f"[auto-repaired] {issue}")


def validate_and_repair(
    result: dict,
    trigger_kind: str,
    already_sent_bodies: Optional[list] = None,
    category_slug: str = "",
) -> tuple[dict, ValidationResult]:
    """
    Validates and auto-repairs a composed message dict.
    Returns (repaired_result, validation_result).
    """
    vr = ValidationResult()
    body = result.get("body", "").strip()
    cta = result.get("cta", "open_ended")

    # 1. Empty body
    if not body:
        vr.fail("Empty body")
        return result, vr

    # 2. Too short
    if len(body) < MIN_BODY_LENGTH:
        vr.fail(f"Body too short ({len(body)} chars, min {MIN_BODY_LENGTH})")
        return result, vr

    # 3. Too long — truncate gracefully at last sentence boundary
    if len(body) > MAX_BODY_LENGTH:
        truncated = body[:MAX_BODY_LENGTH]
        # Find last sentence end
        last_end = max(
            truncated.rfind("."),
            truncated.rfind("?"),
            truncated.rfind("!"),
        )
        if last_end > MAX_BODY_LENGTH // 2:
            body = truncated[:last_end + 1]
            result["body"] = body
            vr.repair(f"Truncated body from {len(result['body'])} to {len(body)} chars")

    # 4. Verbatim repeat
    if already_sent_bodies:
        body_lower = body.lower()
        for prev in already_sent_bodies:
            if prev.strip().lower() == body_lower:
                vr.fail("Body is verbatim repeat of previously sent message")
                return result, vr

    # 5. Specificity — must contain at least one number/price/%
    if not _has_specificity(body):
        vr.fail("Body lacks specificity (no numbers, prices, or percentages found)")
        # Don't return — this is a soft failure, let retry handle it

    # 6. Hallucination markers
    if _has_hallucination_marker(body):
        vr.fail("Body contains hallucination markers")
        return result, vr

    # 6b. URL check — hard fail, -3 penalty per URL
    if _has_url(body):
        cleaned = _strip_urls(body)
        if cleaned and len(cleaned) >= MIN_BODY_LENGTH:
            result["body"] = cleaned
            body = cleaned
            vr.repair("Stripped URLs from body (URLs cause -3 penalty)")
        else:
            vr.fail("Body contains URLs which cause a -3 penalty and cannot be safely stripped")
            return result, vr

    # 7. Preamble check — auto-repair by stripping preamble sentence
    if _has_preamble(body):
        sentences = re.split(r"(?<=[.!?])\s+", body, maxsplit=1)
        if len(sentences) > 1:
            body = sentences[1].strip()
            result["body"] = body
            vr.repair("Stripped opening preamble sentence")
        else:
            vr.fail("Body starts with preamble and couldn't be repaired")

    # 8. CTA shape
    cta_valid, corrected_cta = _validate_cta(cta, trigger_kind)
    if not cta_valid:
        result["cta"] = corrected_cta
        vr.repair(f"CTA corrected from '{cta}' to '{corrected_cta}' for trigger kind '{trigger_kind}'")

    # 9. Medical category — taboo words
    medical_categories = {"dentists", "pharmacies", "doctors"}
    if category_slug in medical_categories:
        taboo_found = [
            w for w in ["cure", "guaranteed", "100% safe", "painless guaranteed"]
            if w.lower() in body.lower()
        ]
        if taboo_found:
            vr.fail(f"Medical taboo words found: {taboo_found}")

    # 10. Multiple CTAs (anti-pattern)
    yes_no_count = len(re.findall(r"\b(YES|STOP|reply yes|reply no|reply 1|reply 2)\b", body, re.IGNORECASE))
    if yes_no_count > 2:
        vr.fail("Multiple CTAs detected in body")

    # Final: if all critical checks passed (only soft failures remain), mark passed
    critical_failures = [i for i in vr.issues if not i.startswith("[auto-repaired]")]
    if not critical_failures:
        vr.passed = True

    return result, vr


def validate_with_retry(
    compose_fn: Callable,
    trigger_kind: str,
    already_sent_bodies: Optional[list] = None,
    category_slug: str = "",
) -> dict:
    """
    Calls compose_fn(), validates, and retries up to MAX_RETRIES times if
    validation fails on critical issues.
    Returns the best result found.
    """
    last_result = None
    last_vr = None

    for attempt in range(1 + MAX_RETRIES):
        result = compose_fn()
        repaired, vr = validate_and_repair(
            result,
            trigger_kind=trigger_kind,
            already_sent_bodies=already_sent_bodies,
            category_slug=category_slug,
        )

        if vr.passed:
            return repaired

        last_result = repaired
        last_vr = vr

        if attempt < MAX_RETRIES:
            # Inject validation issues into the compose_fn on next call
            # (compose_fn should accept an optional hint; if not, just retry)
            pass

    # Return best effort even if validation failed
    return last_result or {}
