"""KG sanity inspection: per-relation degree distribution, hub entities,
cold nodes, multi-hop reach. Read-only — informs the data.py rewrite.

Run: .venv/bin/python3 kg_inspect.py
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
KG = ROOT / "data" / "kg_canonical.csv"
RX = ROOT / "data" / "reviews_30_20.pkl"

USER_RE = re.compile(r"^A[A-Z0-9]{27}$")
ASIN_RE = re.compile(r"^[A-Z0-9]{10}$")


def head_type(s: str) -> str:
    if USER_RE.match(s):
        return "user"
    if ASIN_RE.match(s):
        return "item"
    return "entity"


def quantiles(arr: np.ndarray) -> str:
    if len(arr) == 0:
        return "n/a"
    q = np.quantile(arr, [0.5, 0.9, 0.99])
    return f"med={int(q[0])}  p90={int(q[1])}  p99={int(q[2])}  max={int(arr.max())}"


def main() -> None:
    print(f"Loading {KG.name}")
    kg = pd.read_csv(KG)
    print(f"  {len(kg):,} canonical triples\n")

    rx = pd.read_pickle(RX)
    rx_items = set(rx["asin"].astype(str).unique())
    rx_users = set(rx["user_id"].astype(str).unique())
    print(f"Loading {RX.name}: {len(rx_users)} users, {len(rx_items)} items\n")

    # Tag head type
    kg["head_type"] = kg["head"].astype(str).map(head_type)

    # =========================================================================
    # 1. Triple breakdown by head type × relation
    # =========================================================================
    print("=" * 72)
    print("1. Triple count by head_type × relation")
    print("=" * 72)
    pivot = kg.groupby(["head_type", "relation"]).size().unstack(fill_value=0)
    pivot["TOTAL"] = pivot.sum(axis=1)
    pivot.loc["TOTAL"] = pivot.sum(axis=0)
    print(pivot.to_string())
    print()

    # =========================================================================
    # 2. Per-relation degree distribution PER HEAD TYPE
    # =========================================================================
    print("=" * 72)
    print("2. Per-relation degree (head → tail count) by head_type")
    print("=" * 72)
    for ht in ["user", "item", "entity"]:
        sub = kg[kg["head_type"] == ht]
        if sub.empty:
            continue
        print(f"\n--- head_type = {ht}  (heads: {sub['head'].nunique():,}) ---")
        for rel in sorted(sub["relation"].unique()):
            sub_r = sub[sub["relation"] == rel]
            deg = sub_r.groupby("head").size().values
            n_with = len(deg)
            print(f"  {rel:<14s}  heads_with_edge={n_with:>6,d}  {quantiles(deg)}")

    # =========================================================================
    # 3. Cold nodes: items / users that DON'T appear in KG at all
    # =========================================================================
    print()
    print("=" * 72)
    print("3. Cold-node check (vs reviews_30_20.pkl)")
    print("=" * 72)
    kg_items = set(kg.loc[kg["head_type"] == "item", "head"]) | set(
        kg.loc[kg["tail"].astype(str).map(lambda x: bool(ASIN_RE.match(x))), "tail"]
    )
    kg_users = set(kg.loc[kg["head_type"] == "user", "head"])
    print(f"  items in reviews not in KG: {len(rx_items - kg_items):>4d}  / {len(rx_items)}")
    print(f"  users in reviews not in KG: {len(rx_users - kg_users):>4d}  / {len(rx_users)}")
    print(f"  (NOTE: items only appear as head, never as tail — pure source nodes)")
    print()
    # Per-relation cold items/users (items/users with NO edge of that relation)
    print("  Per-relation coverage among reviews users/items:")
    for ht, ref in [("user", rx_users), ("item", rx_items)]:
        for rel in sorted(kg["relation"].unique()):
            sub = kg[(kg["head_type"] == ht) & (kg["relation"] == rel)]
            covered = ref & set(sub["head"])
            if not covered and ht == "item" and rel.endswith("PREF"):
                continue   # skip noise (items don't have user-side prefs)
            if not covered and ht == "user" and rel in {"HAS_PROPERTY", "DEPICTS", "SET_IN"}:
                continue   # skip noise (users don't have item-side props)
            pct = len(covered) / len(ref)
            bar = "█" * int(pct * 30)
            print(f"    {ht:<6s} × {rel:<14s} {len(covered):>5,}/{len(ref):<5,} ({pct:>5.1%}) {bar}")

    # =========================================================================
    # 4. Hub-entity detection: tail entities with very high in-degree
    # =========================================================================
    print()
    print("=" * 72)
    print("4. Hub entities (high tail in-degree → suspected noise / generic)")
    print("=" * 72)
    tail_deg = kg.groupby("tail").size().sort_values(ascending=False)
    print(f"  total unique tails: {len(tail_deg):,}")
    print(f"  top-20 hubs:")
    for tail, deg in tail_deg.head(20).items():
        rels = kg[kg["tail"] == tail]["relation"].value_counts().head(3).to_dict()
        rels_str = ", ".join(f"{r}:{c}" for r, c in rels.items())
        print(f"    {tail:<40s} {deg:>5,}  ({rels_str})")
    print()
    print(f"  tail in-degree distribution:")
    deg_arr = tail_deg.values
    for cutoff in [1, 2, 5, 10, 50, 100, 500]:
        n = (deg_arr <= cutoff).sum()
        print(f"    deg ≤ {cutoff:>4d}: {n:>6,d} entities ({n/len(deg_arr):>5.1%})")

    # =========================================================================
    # 5. Multi-hop reach from items (item → entity → entity)
    # =========================================================================
    print()
    print("=" * 72)
    print("5. Multi-hop reach: item → entity → entity")
    print("=" * 72)
    # Build adjacency: head → list of tails (any relation)
    adj_out = defaultdict(set)
    for h, t in zip(kg["head"].values, kg["tail"].values):
        adj_out[h].add(t)

    # 1-hop entity neighbors of each item
    item_1hop = defaultdict(set)
    for item in (rx_items & set(kg["head"])):
        for t in adj_out.get(item, set()):
            if not ASIN_RE.match(t) and not USER_RE.match(t):
                item_1hop[item].add(t)

    one_hop_sizes = np.array([len(v) for v in item_1hop.values()])
    print(f"  items with ≥1 1-hop entity: {len(item_1hop):,}")
    print(f"  1-hop entity count: {quantiles(one_hop_sizes)}")

    # 2-hop reach: from item → entity → other entity
    item_2hop = {}
    for item, ents in item_1hop.items():
        reach = set()
        for e in ents:
            reach.update(adj_out.get(e, set()))
        reach -= ents             # exclude direct
        reach.discard(item)
        item_2hop[item] = reach
    two_hop_sizes = np.array([len(v) for v in item_2hop.values()])
    print(f"  2-hop entity reach: {quantiles(two_hop_sizes)}")
    print(f"  items with ≥1 2-hop entity: {(two_hop_sizes > 0).sum():,}")

    # =========================================================================
    # 6. Suggested entity prune threshold
    # =========================================================================
    print()
    print("=" * 72)
    print("6. Suggested entity-prune thresholds (drop low-degree tails)")
    print("=" * 72)
    print(f"  current: {len(tail_deg):,} unique tail entities")
    for cutoff in [1, 2, 3, 5]:
        keep = (tail_deg >= cutoff).sum()
        kept_edges = kg[kg["tail"].isin(tail_deg[tail_deg >= cutoff].index)].shape[0]
        print(
            f"  prune deg < {cutoff}:  keep {keep:>6,d} entities ({keep/len(tail_deg):.1%}),  "
            f"keep {kept_edges:>6,d} edges ({kept_edges/len(kg):.1%})"
        )


if __name__ == "__main__":
    main()
