"""
Pre-computed merchant intelligence layer.

Derives rich, specific insights from raw context before passing to the LLM.
This is what makes messages feel data-driven rather than generic —
the LLM gets pre-digested facts it can anchor on directly.

Examples of what this produces:
  - "CTR is 30% below peer median (2.1% vs 3.0%)"
  - "78 lapsed patients × avg ₹350 = ₹27,300 in recoverable revenue"
  - "Last Google post was 22 days ago (peer best-practice: every 7 days)"
  - "Seasonal relevance: exam-stress bruxism spike (Nov-Feb) — you are in this window"
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional
import math


# ── CTR analysis ──────────────────────────────────────────────────────────────

def ctr_vs_peer(merchant: dict, category: dict) -> dict:
    """
    Returns a dict describing how the merchant's CTR compares to the peer median.
    """
    perf = merchant.get("performance", {})
    peer = category.get("peer_stats", {})

    merchant_ctr = perf.get("ctr") or perf.get("ctr_30d")
    peer_ctr = peer.get("avg_ctr")

    if not merchant_ctr or not peer_ctr:
        return {}

    gap_pct = ((peer_ctr - merchant_ctr) / peer_ctr) * 100
    direction = "below" if gap_pct > 0 else "above"
    gap_abs = abs(gap_pct)

    # Predictive ROI projection — dynamic math, not hardcoded
    views_30d = perf.get("views_30d") or perf.get("views", 1000)

    # Dynamic conversion rate: derive from merchant's own data if available
    cust_agg = merchant.get("customer_aggregate", {})
    active_customers = cust_agg.get("active_count", 0)
    total_customers = cust_agg.get("total_count", 0) or active_customers
    calls_30d = perf.get("calls_30d") or perf.get("calls", 0)
    leads_30d = perf.get("leads_30d") or perf.get("leads", 0)

    if total_customers and views_30d and views_30d > 0:
        # Best signal: actual customers / actual views (lifetime-normalized to 30d)
        conversion_rate = min(total_customers / max(views_30d * 6, 1), 0.25)  # cap at 25%
    elif (calls_30d + leads_30d) and views_30d > 0:
        # Fallback: (calls + leads) / views as a proxy
        conversion_rate = min((calls_30d + leads_30d) / views_30d, 0.25)
    else:
        conversion_rate = category.get("conversion_rate", 0.10)  # category or 10% default
    conversion_rate = max(conversion_rate, 0.03)  # floor at 3%
    conversion_pct = round(conversion_rate * 100, 1)

    # Average transaction: use merchant's own active offers first, then catalog
    merchant_offers = [o for o in merchant.get("offers", []) if o.get("status") == "active"]
    catalog = category.get("offer_catalog", [])
    price_sources = merchant_offers if merchant_offers else catalog
    prices = []
    for offer in price_sources:
        val = offer.get("value") or offer.get("price")
        if val:
            try:
                prices.append(float(str(val).replace("₹", "").replace(",", "").strip()))
            except ValueError:
                pass
    avg_price = sum(prices) / len(prices) if prices else 350

    extra_clicks = 0
    extra_bookings = 0
    extra_revenue = 0

    if gap_pct > 0:  # merchant is below peer
        extra_clicks = (peer_ctr - merchant_ctr) * views_30d
        extra_bookings = extra_clicks * conversion_rate
        extra_revenue = extra_bookings * avg_price

    return {
        "merchant_ctr": round(merchant_ctr * 100, 2),
        "peer_ctr": round(peer_ctr * 100, 2),
        "gap_direction": direction,
        "gap_pct": round(gap_abs, 1),
        "projected_bookings": math.ceil(extra_bookings),
        "projected_revenue": round(extra_revenue),
        "conversion_rate_used": conversion_pct,
        "avg_ticket_used": round(avg_price),
        "summary": (
            f"CTR {round(merchant_ctr*100,1)}% is {round(gap_abs,0):.0f}% {direction} "
            f"the peer median of {round(peer_ctr*100,1)}%. "
            f"Closing this gap → +{math.ceil(extra_bookings)} bookings "
            f"(₹{round(extra_revenue):,}) at your {conversion_pct}% conversion × ₹{round(avg_price)} avg ticket."
            if extra_bookings > 0 else
            f"CTR {round(merchant_ctr*100,1)}% is {round(gap_abs,0):.0f}% {direction} "
            f"the peer median of {round(peer_ctr*100,1)}%"
        ),
        "is_below_peer": gap_pct > 0,
    }


# ── Lapsed revenue opportunity ────────────────────────────────────────────────

def lapsed_revenue_opportunity(merchant: dict, category: dict) -> dict:
    """
    Estimates recoverable revenue from lapsed customers.
    Uses average offer price from category catalog as proxy for avg transaction value.
    """
    agg = merchant.get("customer_aggregate", {})
    lapsed_count = agg.get("lapsed_180d_plus") or agg.get("lapsed_count", 0)
    if not lapsed_count:
        return {}

    # Estimate avg transaction from category offer catalog
    catalog = category.get("offer_catalog", [])
    prices = []
    for offer in catalog:
        val = offer.get("value") or offer.get("price")
        if val:
            try:
                prices.append(float(str(val).replace("₹", "").replace(",", "").strip()))
            except ValueError:
                pass

    avg_price = sum(prices) / len(prices) if prices else 350  # ₹350 default
    opportunity = lapsed_count * avg_price

    return {
        "lapsed_count": lapsed_count,
        "avg_transaction_inr": round(avg_price),
        "opportunity_inr": round(opportunity),
        "summary": (
            f"{lapsed_count} lapsed customers × avg ₹{round(avg_price)} "
            f"= ₹{round(opportunity):,} in recoverable revenue"
        ),
    }


# ── Stale content detection ───────────────────────────────────────────────────

def stale_content_analysis(merchant: dict) -> dict:
    """
    Detects how stale the merchant's Google profile content is.
    """
    signals = merchant.get("signals", [])
    stale_days = None

    for sig in signals:
        sig_str = str(sig)
        if "stale_posts" in sig_str:
            # Extract days from format like "stale_posts:22d"
            parts = sig_str.split(":")
            if len(parts) > 1:
                try:
                    stale_days = int(parts[1].replace("d", "").strip())
                except ValueError:
                    stale_days = 14  # default if we can't parse
            else:
                stale_days = 14
            break

    if stale_days is None:
        return {}

    urgency = "high" if stale_days > 21 else "medium" if stale_days > 10 else "low"
    return {
        "days_since_last_post": stale_days,
        "urgency": urgency,
        "summary": f"Last Google post was {stale_days} days ago (best practice: every 7 days)",
    }


# ── Seasonal relevance ────────────────────────────────────────────────────────

def seasonal_relevance(category: dict, now: Optional[datetime] = None) -> dict:
    """
    Returns seasonal beats that are currently active.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    month = now.month
    month_name = now.strftime("%b")  # e.g. "Apr"

    beats = category.get("seasonal_beats", [])
    active = []

    for beat in beats:
        month_range = beat.get("month_range", beat.get("months", ""))
        note = beat.get("note", "")
        if not month_range:
            continue

        # Parse ranges like "Nov-Feb", "Oct-Dec", "Jun-Aug"
        month_names = [
            "Jan","Feb","Mar","Apr","May","Jun",
            "Jul","Aug","Sep","Oct","Nov","Dec"
        ]
        try:
            parts = month_range.split("-")
            start_m = month_names.index(parts[0].strip()[:3]) + 1
            end_m = month_names.index(parts[1].strip()[:3]) + 1

            # Handle wrap-around (e.g. Nov-Feb)
            if start_m <= end_m:
                in_range = start_m <= month <= end_m
            else:
                in_range = month >= start_m or month <= end_m

            if in_range:
                active.append({"range": month_range, "note": note})
        except (ValueError, IndexError):
            pass

    return {"active_beats": active} if active else {}


# ── Offer analysis ────────────────────────────────────────────────────────────

def best_offer(merchant: dict, category: dict) -> dict:
    """
    Returns the most compelling offer to highlight.
    Prefers active merchant offers with a price over generic catalog offers.
    """
    # Check merchant's active offers first
    merchant_offers = [
        o for o in merchant.get("offers", [])
        if o.get("status") == "active" and o.get("title")
    ]
    if merchant_offers:
        offer = merchant_offers[0]
        return {
            "title": offer.get("title"),
            "source": "merchant_catalog",
            "summary": f"Active offer: {offer.get('title')}",
        }

    # Fall back to category catalog
    catalog = category.get("offer_catalog", [])
    if catalog:
        offer = catalog[0]
        return {
            "title": offer.get("title"),
            "source": "category_catalog",
            "summary": f"Category standard offer: {offer.get('title')}",
        }

    return {}


# ── Review signal ─────────────────────────────────────────────────────────────

def review_signal(merchant: dict, category: dict) -> dict:
    """
    Compares merchant's review count and rating to peer stats.
    """
    perf = merchant.get("performance", {})
    peer = category.get("peer_stats", {})

    rating = perf.get("rating") or merchant.get("identity", {}).get("rating")
    review_count = perf.get("reviews") or merchant.get("identity", {}).get("review_count")
    peer_avg_reviews = peer.get("avg_reviews")
    peer_avg_rating = peer.get("avg_rating")

    result = {}
    if rating and peer_avg_rating:
        diff = round(rating - peer_avg_rating, 1)
        direction = "above" if diff >= 0 else "below"
        result["rating_vs_peer"] = f"{rating}★ ({direction} peer avg {peer_avg_rating}★)"

    if review_count and peer_avg_reviews:
        diff = review_count - peer_avg_reviews
        direction = "more" if diff >= 0 else "fewer"
        result["reviews_vs_peer"] = (
            f"{review_count} reviews ({abs(diff)} {direction} than peer avg {peer_avg_reviews})"
        )

    return result


# ── Trend signal ──────────────────────────────────────────────────────────────

def top_trend(category: dict) -> dict:
    """Returns the strongest trend signal for the category."""
    signals = category.get("trend_signals", [])
    if not signals:
        return {}
    # Sort by absolute delta
    def sort_key(s):
        try:
            return abs(float(s.get("delta_yoy", s.get("change_pct", 0))))
        except (TypeError, ValueError):
            return 0
    top = sorted(signals, key=sort_key, reverse=True)
    if top:
        s = top[0]
        delta = s.get("delta_yoy", s.get("change_pct", ""))
        query = s.get("query", "")
        if delta and query:
            pct = round(float(delta) * 100) if abs(float(delta)) <= 10 else round(float(delta))
            return {
                "query": query,
                "change_pct": pct,
                "summary": f'Searches for "{query}" are up {pct}% year-over-year',
            }
    return {}


# ── Patient cohort insight ────────────────────────────────────────────────────

def patient_cohort_insight(merchant: dict) -> dict:
    """
    Extracts high-value patient/customer cohort stats from customer_aggregate.
    These power the gold-standard case study messages (e.g. 'your 124 high-risk adult patients').
    """
    agg = merchant.get("customer_aggregate", {})
    if not agg:
        return {}

    result = {}
    total = agg.get("total_unique_ytd")
    if total:
        result["total_unique_ytd"] = total

    high_risk = agg.get("high_risk_adult_count")
    if high_risk:
        result["high_risk_adult_count"] = high_risk
        result["high_risk_summary"] = f"{high_risk} high-risk adult patients in roster"

    retention = agg.get("retention_6mo_pct")
    if retention:
        result["retention_6mo_pct"] = round(retention * 100) if retention < 1 else retention
        result["retention_summary"] = f"{result['retention_6mo_pct']}% 6-month retention rate"

    lapsed = agg.get("lapsed_180d_plus") or agg.get("lapsed_count", 0)
    if lapsed:
        result["lapsed_180d_plus"] = lapsed

    return result


# ── Review theme insight ──────────────────────────────────────────────────────

def review_theme_insight(merchant: dict) -> dict:
    """
    Surfaces the merchant's most relevant review theme. This is the ONLY
    source of truth for generated review_theme_emerged triggers, which carry
    just a placeholder payload.
    """
    themes = merchant.get("review_themes", [])
    if not themes:
        return {}

    def sort_key(t):
        return (t.get("occurrences_30d", 0), t.get("sentiment") == "neg")

    top = sorted(themes, key=sort_key, reverse=True)[0]
    theme = top.get("theme", "")
    sentiment = top.get("sentiment", "")
    occurrences = top.get("occurrences_30d", 0)
    quote = top.get("common_quote", "")

    summary = f"{occurrences} reviews (30d) mention '{theme}' ({sentiment})"
    if quote:
        summary += f' — e.g. "{quote}"'

    return {
        "theme": theme, "sentiment": sentiment,
        "occurrences_30d": occurrences, "common_quote": quote,
        "summary": summary,
    }


# ── Aggregate all insights ────────────────────────────────────────────────────

def derive_insights(merchant: dict, category: dict, trigger: dict) -> dict:
    """
    Master function. Returns a structured insights dict ready to inject into prompts.
    Each insight is self-contained and phrased as a ready-to-use fact.
    """
    trigger_kind = trigger.get("kind", "")

    insights = {
        "ctr_analysis": ctr_vs_peer(merchant, category),
        "best_offer": best_offer(merchant, category),
        "review_signal": review_signal(merchant, category),
    }

    # Patient/customer cohort stats
    cohort = patient_cohort_insight(merchant)
    if cohort:
        insights["patient_cohort"] = cohort

    # Lapsed revenue — high value for retention/recall triggers
    lapsed = lapsed_revenue_opportunity(merchant, category)
    if lapsed:
        insights["lapsed_revenue"] = lapsed

    # Stale content — relevant for profile/GBP triggers
    stale = stale_content_analysis(merchant)
    if stale:
        insights["stale_content"] = stale

    # Seasonal beats — always include if active
    seasonal = seasonal_relevance(category)
    if seasonal.get("active_beats"):
        insights["seasonal"] = seasonal

    # Top trend — relevant for trend/research/competitor triggers
    if any(k in trigger_kind for k in ("trend", "research", "digest", "competitor")):
        trend = top_trend(category)
        if trend:
            insights["top_trend"] = trend

    # Review theme — primary grounding fact for review_theme_emerged triggers
    if "review_theme" in trigger_kind:
        review_theme = review_theme_insight(merchant)
        if review_theme:
            insights["review_theme"] = review_theme

    # Performance delta summary
    perf = merchant.get("performance", {})
    delta = perf.get("delta_7d", {})
    if delta:
        parts = []
        for metric, val in delta.items():
            if val is not None:
                try:
                    pct = round(float(val) * 100)
                    direction = "up" if pct > 0 else "down"
                    parts.append(f"{metric.replace('_pct','')} {direction} {abs(pct)}% (7d)")
                except (TypeError, ValueError):
                    pass
        if parts:
            insights["performance_delta"] = {
                "summary": ", ".join(parts),
                "raw": delta,
            }

    return insights

# ── Merchant DNA Extraction ────────────────────────────────────────────────────

def extract_merchant_dna(merchant: dict, category: dict = None) -> dict:
    """
    Extract a compact DNA profile for the merchant.
    Called during context ingestion (store.store scope=merchant).
    `category` may be passed directly; if not provided, we build without it.
    """
    dna = {}

    # 1. Owner Address
    identity = merchant.get("identity", {})
    category_slug = merchant.get("category_slug", "")
    name = identity.get("owner_first_name") or identity.get("name", "there")
    if category_slug == "dentists" and not name.startswith("Dr."):
        dna["owner_address"] = f"Dr. {name}"
    else:
        dna["owner_address"] = name

    # 2. Language Mode
    languages = identity.get("languages", ["en"])
    dna["language_mode"] = "hinglish" if "hi" in languages else "english"

    # 3. Category Voice
    voice_map = {
        "dentists": "clinical-peer",
        "pharmacies": "clinical-peer",
        "salons": "sensory-visual",
        "gyms": "energetic-transformational",
        "restaurants": "warm-local"
    }
    dna["category_voice"] = voice_map.get(category_slug, "professional")

    # 4. Top Offer (from merchant's own offers)
    active_offers = [o for o in merchant.get("offers", []) if o.get("status") == "active"]
    if active_offers:
        o = active_offers[0]
        price = o.get("price") or o.get("value") or ""
        dna["top_offer"] = f"{o.get('title', '')} @ ₹{price}" if price else o.get("title", "")
    else:
        dna["top_offer"] = ""

    # 5. Best Hook (CTR vs peer) — needs category, skip gracefully if unavailable
    if category:
        ctr_info = ctr_vs_peer(merchant, category)
        if ctr_info:
            dna["best_hook"] = (
                f"CTR {ctr_info.get('merchant_ctr')}% vs peer {ctr_info.get('peer_ctr')}% "
                f"= {ctr_info.get('gap_pct')}% {ctr_info.get('gap_direction')}"
            )
        else:
            dna["best_hook"] = ""

        # 6. Urgency Trigger (Lapsed Revenue)
        lapsed = lapsed_revenue_opportunity(merchant, category)
        dna["urgency_trigger"] = lapsed.get("summary", "") if lapsed else ""

        # 7. Seasonal Window
        seasonal = seasonal_relevance(category)
        if seasonal and seasonal.get("active_beats"):
            beat = seasonal["active_beats"][0]
            dna["seasonal_window"] = f"active: {beat.get('note')} ({beat.get('range')})"
        else:
            dna["seasonal_window"] = "none"
    else:
        dna["best_hook"] = ""
        dna["urgency_trigger"] = ""
        dna["seasonal_window"] = "none"

    return dna