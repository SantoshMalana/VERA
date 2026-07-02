"""
submission_cache.py — Version-aware fallback cache
=====================================================================
Cache is a FALLBACK ONLY. select_and_compose() always tries a live,
grounded compose() first. This is only consulted if that call raises,
and only served if the merchant/trigger context hasn't moved on since
the cached response was generated (stale cache is worse than no cache).
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_CACHE_PATH = Path(__file__).parent / "submission_cache.json"
_cache: dict[str, dict] = {}
_loaded = False


def _load_cache() -> None:
    global _cache, _loaded
    if _loaded:
        return
    if _CACHE_PATH.exists():
        try:
            with open(_CACHE_PATH, "r", encoding="utf-8") as f:
                _cache = json.load(f)
            logger.info("Loaded %d cached responses from submission_cache.json", len(_cache))
        except Exception as exc:
            logger.warning("Failed to load submission cache: %s", exc)
            _cache = {}
    _loaded = True


def _cache_key(trigger_id: str, merchant_id: str) -> str:
    return f"{trigger_id}::{merchant_id}"


def get_cached_response(
    trigger_id: str,
    merchant_id: str,
    merchant_version: Optional[int] = None,
    trigger_version: Optional[int] = None,
) -> Optional[dict]:
    _load_cache()
    key = _cache_key(trigger_id, merchant_id)
    entry = _cache.get(key)
    if not entry and merchant_id:
        fallback_key = _cache_key(trigger_id, "")
        entry = _cache.get(fallback_key)
        if entry:
            logger.info("Serving fallback cached response for trigger=%s (city-scope)", trigger_id)
    if not entry:
        return None

    cached_mv = entry.get("_merchant_version")
    cached_tv = entry.get("_trigger_version")
    if merchant_version is not None and cached_mv is not None and cached_mv != merchant_version:
        logger.info("Cache STALE (merchant v%s != current v%s) for %s — skipping", cached_mv, merchant_version, key)
        return None
    if trigger_version is not None and cached_tv is not None and cached_tv != trigger_version:
        logger.info("Cache STALE (trigger v%s != current v%s) for %s — skipping", cached_tv, trigger_version, key)
        return None

    logger.info("Serving cached fallback for trigger=%s merchant=%s", trigger_id, merchant_id)
    return entry


def save_response(
    trigger_id: str, merchant_id: str, action: dict,
    merchant_version: Optional[int] = None, trigger_version: Optional[int] = None,
) -> None:
    _load_cache()
    key = _cache_key(trigger_id, merchant_id)
    action = dict(action)
    action["_merchant_version"] = merchant_version
    action["_trigger_version"] = trigger_version
    _cache[key] = action
    try:
        with open(_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_cache, f, ensure_ascii=False, indent=2)
        logger.info("Saved response to cache: %s", key)
    except Exception as exc:
        logger.error("Failed to save to cache: %s", exc)


def list_cached_keys() -> list[str]:
    _load_cache()
    return list(_cache.keys())
