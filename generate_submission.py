"""
Offline submission.jsonl generator.

Reads the expanded dataset, runs compose() for each of the 30 canonical test pairs,
and writes submission.jsonl (one JSON line per test pair).

Usage:
  python generate_submission.py \
    --dataset ../magicpin-challenge/dataset/expanded \
    --out submission.jsonl
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from composer import compose_message


def load_json(path: Path) -> dict | list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="../magicpin-challenge/dataset/expanded",
        help="Path to the expanded dataset directory",
    )
    parser.add_argument("--out", default="submission.jsonl")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        print(f"ERROR: Dataset directory not found: {dataset_dir}")
        print("Run: python3 dataset/generate_dataset.py first")
        sys.exit(1)

    # Load test pairs
    test_pairs_path = dataset_dir / "test_pairs.json"
    if not test_pairs_path.exists():
        # Try alternate location
        alt = dataset_dir.parent / "test_pairs.json"
        if alt.exists():
            test_pairs_path = alt
        else:
            print(f"ERROR: test_pairs.json not found in {dataset_dir}")
            print("Expected: expanded/test_pairs.json")
            sys.exit(1)

    test_pairs = load_json(test_pairs_path)
    print(f"Loaded {len(test_pairs)} test pairs")

    # Load all categories
    categories: dict[str, dict] = {}
    cat_dir = dataset_dir / "categories"
    if not cat_dir.exists():
        cat_dir = dataset_dir.parent / "categories"
    for f in cat_dir.glob("*.json"):
        data = load_json(f)
        slug = data.get("slug", f.stem)
        categories[slug] = data
    print(f"Loaded {len(categories)} categories: {list(categories.keys())}")

    # Load all merchants
    merchants: dict[str, dict] = {}
    m_dir = dataset_dir / "merchants"
    if not m_dir.exists():
        m_dir = dataset_dir
    for f in m_dir.glob("m_*.json"):
        data = load_json(f)
        mid = data.get("merchant_id", f.stem)
        merchants[mid] = data
    print(f"Loaded {len(merchants)} merchants")

    # Load all customers
    customers: dict[str, dict] = {}
    c_dir = dataset_dir / "customers"
    if not c_dir.exists():
        c_dir = dataset_dir
    for f in c_dir.glob("c_*.json"):
        data = load_json(f)
        cid = data.get("customer_id", f.stem)
        customers[cid] = data
    print(f"Loaded {len(customers)} customers")

    # Load all triggers
    triggers: dict[str, dict] = {}
    t_dir = dataset_dir / "triggers"
    if not t_dir.exists():
        t_dir = dataset_dir
    for f in t_dir.glob("trg_*.json"):
        data = load_json(f)
        tid = data.get("id", f.stem)
        triggers[tid] = data
    print(f"Loaded {len(triggers)} triggers")

    # Generate messages for each test pair
    results = []
    errors = 0

    for i, pair in enumerate(test_pairs, 1):
        test_id = pair.get("test_id", f"T{i:02d}")
        merchant_id = pair.get("merchant_id")
        trigger_id = pair.get("trigger_id")
        customer_id = pair.get("customer_id")

        print(f"\n[{i}/{len(test_pairs)}] {test_id}: {merchant_id} + {trigger_id}", end=" ... ")

        merchant = merchants.get(merchant_id)
        trigger = triggers.get(trigger_id)
        customer = customers.get(customer_id) if customer_id else None

        if not merchant:
            print(f"SKIP — merchant {merchant_id} not found")
            errors += 1
            continue
        if not trigger:
            print(f"SKIP — trigger {trigger_id} not found")
            errors += 1
            continue

        category_slug = merchant.get("category_slug", "")
        category = categories.get(category_slug)
        if not category:
            print(f"SKIP — category {category_slug} not found")
            errors += 1
            continue

        try:
            result = compose_message(category, merchant, trigger, customer)
            row = {
                "test_id": test_id,
                "merchant_id": merchant_id,
                "trigger_id": trigger_id,
                "customer_id": customer_id,
                "body": result.get("body", ""),
                "cta": result.get("cta", "open_ended"),
                "send_as": result.get("send_as", "vera"),
                "suppression_key": trigger.get("suppression_key", ""),
                "rationale": result.get("rationale", ""),
            }
            results.append(row)
            print("OK")
            print(f"  → {row['body'][:100]}...")
        except Exception as exc:
            print(f"ERROR: {exc}")
            errors += 1

    # Write output
    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"Wrote {len(results)} lines to {out_path}")
    if errors:
        print(f"WARNING: {errors} pairs failed")
    print("Done.")


if __name__ == "__main__":
    main()
