"""KG canonicalisation: collapse 2,687 raw relations to ~9 canonical classes,
normalise entity strings, and write a NEW csv (does NOT touch the source file).

Inputs (read-only):
    data/df_edges_all_aspect1.csv        — raw KG (head, tail, relation)

Outputs (new):
    data/kg_canonical.csv                — cleaned KG with canonical relation
    data/kg_relation_map.json            — full raw→canonical mapping for audit
    data/kg_canonical_report.txt         — before/after stats

Usage:
    .venv/bin/python3 kg_clean.py
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "data" / "df_edges_all_aspect1.csv"
OUT_CSV = ROOT / "data" / "kg_canonical.csv"
OUT_MAP = ROOT / "data" / "kg_relation_map.json"
OUT_REPORT = ROOT / "data" / "kg_canonical_report.txt"

# ---------------------------------------------------------------------------
# Canonical relation buckets
# ---------------------------------------------------------------------------
# Designed from the top-50 raw relations (covering ~89% of edges).
# Anything not listed here is mapped to OTHER.

RELATION_MAP: dict[str, str] = {
    # ----- user-side: positive preference -----
    "prefers": "POSITIVE_PREF",
    "appreciates": "POSITIVE_PREF",
    "enjoys": "POSITIVE_PREF",
    "loves": "POSITIVE_PREF",
    "values": "POSITIVE_PREF",
    "admires": "POSITIVE_PREF",
    "enjoyed": "POSITIVE_PREF",
    "likes": "POSITIVE_PREF",
    "attracted_to": "POSITIVE_PREF",
    "appreciated": "POSITIVE_PREF",
    "prioritizes": "POSITIVE_PREF",
    "seeks": "POSITIVE_PREF",
    "expects": "POSITIVE_PREF",
    "is_a_fan_of": "POSITIVE_PREF",
    "identifies_with": "POSITIVE_PREF",
    "fascinated_by": "POSITIVE_PREF",
    "is_attracted_to": "POSITIVE_PREF",
    "desires": "POSITIVE_PREF",
    "finds_intriguing": "POSITIVE_PREF",
    "expects_high": "POSITIVE_PREF",
    "finds_interesting": "POSITIVE_PREF",
    "loved": "POSITIVE_PREF",
    "intrigued_by": "POSITIVE_PREF",
    "wishes_for": "POSITIVE_PREF",

    # ----- user-side: interest (softer than love/prefer; kept separate) -----
    "interested_in": "INTERESTED_IN",
    "is_interested_in": "INTERESTED_IN",
    "open_to": "INTERESTED_IN",

    # ----- user-side: negative preference -----
    "dislikes": "NEGATIVE_PREF",
    "critical_of": "NEGATIVE_PREF",
    "criticizes": "NEGATIVE_PREF",
    "disinterested_in": "NEGATIVE_PREF",
    "lacks": "NEGATIVE_PREF",
    "less_concerned_with": "NEGATIVE_PREF",
    "disappointed_in": "NEGATIVE_PREF",
    "skeptical_of": "NEGATIVE_PREF",

    # ----- user-side: neutral / weak preference -----
    "tolerates": "NEUTRAL_PREF",
    "neutral": "NEUTRAL_PREF",
    "can_tolerate": "NEUTRAL_PREF",
    "can_handle": "NEUTRAL_PREF",

    # ----- item-side: structural / functional features -----
    "features": "HAS_PROPERTY",
    "has": "HAS_PROPERTY",
    "offers": "HAS_PROPERTY",
    "provides": "HAS_PROPERTY",
    "includes": "HAS_PROPERTY",
    "facilitates": "HAS_PROPERTY",
    "supports": "HAS_PROPERTY",
    "delivers": "HAS_PROPERTY",
    "requires": "HAS_PROPERTY",
    "enhances": "HAS_PROPERTY",
    "utilizes": "HAS_PROPERTY",
    "contains": "HAS_PROPERTY",

    # ----- item-side: thematic content -----
    "depicts": "DEPICTS",
    "addresses": "DEPICTS",
    "presents": "DEPICTS",
    "showcases": "DEPICTS",
    "focuses_on": "DEPICTS",
    "centers_around": "DEPICTS",
    "explores": "DEPICTS",
    "portrays": "DEPICTS",
    "represents": "DEPICTS",
    "describes": "DEPICTS",
    "is_about": "DEPICTS",
    "emphasizes": "DEPICTS",

    # ----- entity taxonomy -----
    "is_a": "IS_A",
    "is_a_type_of": "IS_A",
    "is_a_part_of": "IS_A",
    "belongs_to": "IS_A",
    "is": "IS_A",
    "is_a_component_of": "IS_A",
    "is_part_of": "IS_A",
    "is_a_feature_of": "IS_A",
    "is_aspect_of": "IS_A",
    "is_a_genre_of": "IS_A",
    "is_an_example_of": "IS_A",
    "theme": "IS_A",

    # ----- spatio-temporal setting -----
    "set_in": "SET_IN",
    "is_set_in": "SET_IN",

    # ----- entity-entity causality -----
    "influences": "INFLUENCES",
    "causes": "INFLUENCES",
    "contributes_to": "INFLUENCES",
    "affects": "INFLUENCES",
    "provides_context_for": "INFLUENCES",
    "develops": "INFLUENCES",
}

OTHER = "OTHER"

# ---------------------------------------------------------------------------
# Entity-string normalisation
# ---------------------------------------------------------------------------
# user_id: 28-char string starting with 'A' (Amazon convention)
# asin:    10-char alphanumeric uppercase (Amazon convention)
USER_RE = re.compile(r"^A[A-Z0-9]{27}$")
ASIN_RE = re.compile(r"^[A-Z0-9]{10}$")


def is_id(s: str) -> bool:
    return bool(USER_RE.match(s) or ASIN_RE.match(s))


def normalise_entity(s: str) -> str:
    """Lowercase + collapse whitespace to single underscore. IDs untouched."""
    if is_id(s):
        return s
    s = s.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Reading {SRC}")
    df = pd.read_csv(SRC, index_col=0)
    df.columns = ["head", "tail", "relation"]
    n_raw = len(df)
    print(f"  {n_raw:,} raw triples")

    # --- Stats: BEFORE ---
    raw_rel_counts = df["relation"].value_counts()
    n_raw_relations = len(raw_rel_counts)
    n_raw_entities = len(set(df["head"]) | set(df["tail"]))

    # --- Apply canonical mapping ---
    df["relation_canonical"] = df["relation"].map(RELATION_MAP).fillna(OTHER)

    # --- Normalise entities (skip IDs) ---
    df["head"] = df["head"].astype(str).map(normalise_entity)
    df["tail"] = df["tail"].astype(str).map(normalise_entity)

    # --- Drop self-loops created by normalisation collisions ---
    self_loops = (df["head"] == df["tail"]).sum()
    if self_loops:
        print(f"  dropping {self_loops} self-loops introduced by string normalisation")
        df = df[df["head"] != df["tail"]].copy()

    # --- Drop exact-duplicate triples (post-normalisation) ---
    pre_dedup = len(df)
    df = df.drop_duplicates(subset=["head", "tail", "relation_canonical"])
    n_dedup = pre_dedup - len(df)
    if n_dedup:
        print(f"  dropping {n_dedup:,} duplicate triples after normalisation")

    # --- Stats: AFTER ---
    canon_rel_counts = df["relation_canonical"].value_counts()
    n_canon_entities = len(set(df["head"]) | set(df["tail"]))

    # --- Write outputs ---
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out = df[["head", "tail", "relation_canonical", "relation"]].rename(
        columns={"relation_canonical": "relation", "relation": "relation_raw"}
    )
    df_out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}  ({len(df_out):,} triples)")

    # Full raw→canonical map for audit (every raw relation we saw)
    full_map = {
        rel: RELATION_MAP.get(rel, OTHER)
        for rel in sorted(raw_rel_counts.index)
    }
    OUT_MAP.write_text(json.dumps(full_map, indent=2, ensure_ascii=False))
    print(f"Wrote {OUT_MAP}  ({len(full_map):,} raw relations)")

    # --- Report ---
    lines = []
    lines.append("=" * 60)
    lines.append("KG canonicalisation report")
    lines.append("=" * 60)
    lines.append(f"Source : {SRC.name}")
    lines.append(f"Output : {OUT_CSV.name}")
    lines.append("")
    lines.append("--- BEFORE ---")
    lines.append(f"  triples          : {n_raw:,}")
    lines.append(f"  unique relations : {n_raw_relations:,}")
    lines.append(f"  unique entities  : {n_raw_entities:,}")
    lines.append("")
    lines.append("--- AFTER  ---")
    lines.append(f"  triples          : {len(df):,}  (dropped {n_raw - len(df):,})")
    lines.append(f"  unique relations : {len(canon_rel_counts):,}")
    lines.append(f"  unique entities  : {n_canon_entities:,}  "
                 f"(merged {n_raw_entities - n_canon_entities:,} via case/space normalisation)")
    lines.append("")
    lines.append("--- Canonical relation distribution ---")
    total = canon_rel_counts.sum()
    for rel, cnt in canon_rel_counts.items():
        lines.append(f"  {rel:<16s} {cnt:>7,}  ({cnt/total:.1%})")
    lines.append("")
    lines.append("--- Top-20 raw relations falling into OTHER ---")
    other_raws = [r for r, c in raw_rel_counts.items() if RELATION_MAP.get(r, OTHER) == OTHER]
    for rel in other_raws[:20]:
        lines.append(f"  {rel:<32s} {raw_rel_counts[rel]:>5,}")
    lines.append(f"  ... ({len(other_raws):,} total raw relations in OTHER)")

    OUT_REPORT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT_REPORT}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
