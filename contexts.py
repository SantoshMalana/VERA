"""
Versioned in-memory context store.
Stores category / merchant / customer / trigger payloads pushed by the judge.
All lookups are O(1); version conflicts return the current version for a 409.
"""
from __future__ import annotations
from threading import Lock
from typing import Any, Optional
import time

class ContextStore:
    def __init__(self) -> None:
        # key: (scope, context_id)  →  {"version": int, "payload": dict}
        self._store: dict[tuple[str, str], dict] = {}
        self._lock = Lock()

    # ------------------------------------------------------------------ write

    def store(self, scope: str, context_id: str, version: int, payload: dict) -> dict:
        """
        Returns {"accepted": True} or {"accepted": False, "current_version": int}.
        Thread-safe.
        """
        key = (scope, context_id)
        with self._lock:
            existing = self._store.get(key)
            if existing and existing["version"] >= version:
                return {"accepted": False, "current_version": existing["version"]}
            
            # --- Merchant DNA Extraction (with category if already stored) ---
            if scope == "merchant":
                from insights import extract_merchant_dna
                category_slug = payload.get("category_slug", "")
                cat_entry = self._store.get(("category", category_slug))
                cat_payload = cat_entry["payload"] if cat_entry else None
                payload["_dna"] = extract_merchant_dna(payload, category=cat_payload)

            self._store[key] = {"version": version, "payload": payload, "updated_at": time.time()}
            return {"accepted": True}

    # ------------------------------------------------------------------ read

    def get(self, scope: str, context_id: str) -> Optional[dict]:
        entry = self._store.get((scope, context_id))
        return entry["payload"] if entry else None

    def get_version(self, scope: str, context_id: str) -> Optional[int]:
        entry = self._store.get((scope, context_id))
        return entry["version"] if entry else None

    def is_recently_updated(self, scope: str, context_id: str, since_seconds: int = 300) -> bool:
        entry = self._store.get((scope, context_id))
        if entry and "updated_at" in entry:
            return (time.time() - entry["updated_at"]) <= since_seconds
        return False

    def get_all(self, scope: str) -> list[dict]:
        return [
            v["payload"]
            for (s, _), v in self._store.items()
            if s == scope
        ]

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
        for (scope, _) in self._store:
            if scope in counts:
                counts[scope] += 1
        return counts

    # ------------------------------------------------------------------ helpers

    def get_merchant_with_category(self, merchant_id: str) -> tuple:
        """Returns (merchant_payload, category_payload)."""
        merchant = self.get("merchant", merchant_id)
        if not merchant:
            return None, None
        category_slug = merchant.get("category_slug", "")
        category = self.get("category", category_slug)
        return merchant, category

    def get_trigger(self, trigger_id: str) -> Optional[dict]:
        return self.get("trigger", trigger_id)

    def get_customer(self, customer_id: str) -> Optional[dict]:
        if not customer_id:
            return None
        return self.get("customer", customer_id)

    def active_triggers(self) -> list[dict]:
        """Returns all stored trigger payloads."""
        return self.get_all("trigger")


# Singleton used across the app
store = ContextStore()