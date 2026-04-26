"""KG loader for the canonicalised KG (data/kg_canonical.csv).

Outputs three relation-typed adjacency dicts (item / user / entity), an
explicit bridge-entity set, and legacy aliases that match the existing
`data.py::build_kg_index` API so the current model.py keeps running.

Read kg_canonical.csv (head, tail, relation, relation_raw), produced by
kg_clean.py.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

log = logging.getLogger(__name__)

# Same regexes as kg_clean.py: skip normalisation for IDs.
USER_RE = re.compile(r"^A[A-Z0-9]{27}$")
ASIN_RE = re.compile(r"^[A-Z0-9]{10}$")


def _is_user(s: str) -> bool:
    return bool(USER_RE.match(s))


def _is_item(s: str) -> bool:
    return bool(ASIN_RE.match(s))


@dataclass
class KGIndexV2:
    # Relation-typed adjacency (head index → list of (relation, tail entity))
    item_kg_adj: Dict[int, List[Tuple[str, str]]] = field(default_factory=dict)
    user_kg_adj: Dict[int, List[Tuple[str, str]]] = field(default_factory=dict)

    # Bridge = entities that BOTH some item and some user point to.
    # This is the actual cross-side signal.
    bridge_entities: Set[str] = field(default_factory=set)

    # Canonical relations (sorted), and entity vocab
    relations: List[str] = field(default_factory=list)
    entity_to_idx: Dict[str, int] = field(default_factory=dict)

    # Legacy aliases that mirror the existing build_kg_index() return shape.
    # These drop the relation type so model.py / RecDataset can keep using
    # them without modification (sampler.get_kg_neighbor uses these).
    legacy_kg_adj: Dict[int, List[str]] = field(default_factory=dict)
    legacy_kg_rev_adj: Dict[str, List[int]] = field(default_factory=dict)
    legacy_aspect_set: Set[str] = field(default_factory=set)


def build_kg_index_v2(
    kg_path: str,
    asin_to_idx: Dict[str, int],
    user_id_to_idx: Dict[str, int] | None = None,
    *,
    prune_degree: int = 2,
    drop_other: bool = False,
    drop_negative: bool = False,
) -> KGIndexV2:
    """Build a KGIndexV2 from kg_canonical.csv.

    Args:
        kg_path: path to kg_canonical.csv (must have head, tail, relation columns)
        asin_to_idx: maps asin string → item_idx (from data.py::load_interactions)
        user_id_to_idx: maps user_id string → user_idx (optional; user-side KG
                        skipped if None)
        prune_degree: drop tail entities with global degree < this. 2 by default
                      removes the ~11.8k singleton entities (68.9% of vocab,
                      only 17% of edges).
        drop_other: if True, drop the OTHER relation bucket (12% of edges, mostly
                    semantically vague).
        drop_negative: if True, drop NEGATIVE_PREF / NEUTRAL_PREF from user side.
                       Some KG-aware models only use positive signals.

    Returns:
        KGIndexV2 with relation-typed and legacy-shaped adjacency dicts.
    """
    path = Path(kg_path)
    if not path.exists():
        raise FileNotFoundError(f"KG file not found: {path}")

    log.info("Loading canonical KG from %s", path)
    df = pd.read_csv(path)
    n_raw = len(df)
    expected = {"head", "tail", "relation"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    df["head"] = df["head"].astype(str)
    df["tail"] = df["tail"].astype(str)
    df["relation"] = df["relation"].astype(str)
    log.info("  %d raw triples, %d relations", n_raw, df["relation"].nunique())

    # ---- Optional relation filtering ----
    drop_rels = set()
    if drop_other:
        drop_rels.add("OTHER")
    if drop_negative:
        drop_rels |= {"NEGATIVE_PREF", "NEUTRAL_PREF"}
    if drop_rels:
        before = len(df)
        df = df[~df["relation"].isin(drop_rels)].copy()
        log.info("  dropped relations %s: %d → %d edges", drop_rels, before, len(df))

    # ---- Optional entity prune ----
    if prune_degree > 1:
        tail_deg = df["tail"].value_counts()
        keep_tails = set(tail_deg[tail_deg >= prune_degree].index)
        before = len(df)
        df = df[df["tail"].isin(keep_tails)].copy()
        n_dropped_entities = (tail_deg < prune_degree).sum()
        log.info(
            "  pruned %d singleton entities (deg<%d): %d → %d edges",
            n_dropped_entities, prune_degree, before, len(df),
        )

    # ---- Tag head type ----
    df["head_type"] = df["head"].map(
        lambda s: "user" if _is_user(s) else ("item" if _is_item(s) else "entity")
    )

    # ---- item_kg_adj ----
    item_kg_adj: Dict[int, List[Tuple[str, str]]] = defaultdict(list)
    item_df = df[df["head_type"] == "item"]
    n_item_edges_kept = 0
    for asin, rel, tail in zip(item_df["head"], item_df["relation"], item_df["tail"]):
        idx = asin_to_idx.get(asin)
        if idx is None:
            continue   # KG has 30 items not in interaction set; drop them
        item_kg_adj[idx].append((rel, tail))
        n_item_edges_kept += 1

    # ---- user_kg_adj ----
    user_kg_adj: Dict[int, List[Tuple[str, str]]] = defaultdict(list)
    n_user_edges_kept = 0
    if user_id_to_idx is not None:
        user_df = df[df["head_type"] == "user"]
        for uid, rel, tail in zip(user_df["head"], user_df["relation"], user_df["tail"]):
            idx = user_id_to_idx.get(uid)
            if idx is None:
                continue
            user_kg_adj[idx].append((rel, tail))
            n_user_edges_kept += 1

    # ---- Bridge entities ----
    item_tails = {t for tails in item_kg_adj.values() for _, t in tails}
    user_tails = {t for tails in user_kg_adj.values() for _, t in tails}
    bridge = item_tails & user_tails if user_id_to_idx is not None else item_tails

    # ---- Entity vocab (only entities used by item or user side) ----
    used_entities = sorted(item_tails | user_tails)
    entity_to_idx = {e: i for i, e in enumerate(used_entities)}

    # ---- Legacy aliases (drop relation type) ----
    legacy_kg_adj: Dict[int, List[str]] = {
        idx: [t for _, t in pairs] for idx, pairs in item_kg_adj.items()
    }
    legacy_kg_rev_adj: Dict[str, List[int]] = defaultdict(list)
    for idx, tails in legacy_kg_adj.items():
        for t in tails:
            legacy_kg_rev_adj[t].append(idx)

    relations = sorted(df["relation"].unique())

    log.info(
        "KGIndexV2 built: %d item heads, %d user heads, %d bridge entities, "
        "%d total entities in vocab",
        len(item_kg_adj), len(user_kg_adj), len(bridge), len(entity_to_idx),
    )
    log.info(
        "  item edges kept: %d  |  user edges kept: %d",
        n_item_edges_kept, n_user_edges_kept,
    )

    return KGIndexV2(
        item_kg_adj=dict(item_kg_adj),
        user_kg_adj=dict(user_kg_adj),
        bridge_entities=bridge,
        relations=relations,
        entity_to_idx=entity_to_idx,
        legacy_kg_adj=legacy_kg_adj,
        legacy_kg_rev_adj=dict(legacy_kg_rev_adj),
        legacy_aspect_set=set(used_entities),
    )


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    p = argparse.ArgumentParser()
    p.add_argument("--kg", default="data/kg_canonical.csv")
    p.add_argument("--rx", default="data/reviews_30_20.pkl")
    p.add_argument("--prune", type=int, default=2)
    p.add_argument("--drop-other", action="store_true")
    p.add_argument("--drop-negative", action="store_true")
    args = p.parse_args()

    rx = pd.read_pickle(args.rx)
    rx = rx[rx["like"] == True].copy()
    rx["asin"] = rx["asin"].astype(str)
    rx["user_id"] = rx["user_id"].astype(str)
    asin_to_idx = {a: i for i, a in enumerate(sorted(rx["asin"].unique()))}
    user_id_to_idx = {u: i for i, u in enumerate(sorted(rx["user_id"].unique()))}
    print(f"reviews_30_20 (positive only): {len(asin_to_idx)} items, {len(user_id_to_idx)} users")
    print()

    kg = build_kg_index_v2(
        args.kg, asin_to_idx, user_id_to_idx,
        prune_degree=args.prune,
        drop_other=args.drop_other,
        drop_negative=args.drop_negative,
    )

    print()
    print("=" * 60)
    print("KGIndexV2 summary")
    print("=" * 60)
    print(f"  relations               : {kg.relations}")
    print(f"  items with KG edges     : {len(kg.item_kg_adj)} / {len(asin_to_idx)}")
    print(f"  users with KG edges     : {len(kg.user_kg_adj)} / {len(user_id_to_idx)}")
    print(f"  bridge entities         : {len(kg.bridge_entities):,}")
    print(f"  total entity vocab      : {len(kg.entity_to_idx):,}")
    print(f"  legacy_kg_adj           : {len(kg.legacy_kg_adj)} items")
    print(f"  legacy_kg_rev_adj       : {len(kg.legacy_kg_rev_adj)} entity keys")

    # Sample 3 items and 3 users
    import itertools
    print("\n--- 3 sample items ---")
    for idx, pairs in itertools.islice(kg.item_kg_adj.items(), 3):
        print(f"  item_idx={idx}: {pairs}")
    print("\n--- 3 sample users ---")
    for idx, pairs in itertools.islice(kg.user_kg_adj.items(), 3):
        print(f"  user_idx={idx}: {pairs[:5]} ... ({len(pairs)} total)")
