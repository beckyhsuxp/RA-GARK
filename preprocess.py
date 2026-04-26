"""
preprocess.py — k-core filter for Amazon Books reviews.

Faithful mirror of the original `preprocess.ipynb` pipeline (`filter_reviews`):

    clean_review                       (verified_purchase + dedup + has English letter)
    filter_like_ratio                  (1st pass on full corpus)
    iter_filter_review_num MAX_COUNT=1 (one pass: drop items <I, drop users <U)
    apply split_review_text + drop empty
    iter_filter_review_num MAX_COUNT=10 (full converge)
    filter_like_ratio                  (2nd pass on iter-filtered data)

The MAX_COUNT semantics matter — the original notebook does ONE pass before
sentence splitting, then full convergence after. Replacing the first call
with full convergence cascades to 0 rows on this corpus, so we keep both
calls exactly as the notebook had them.

Usage:

    # reproduce 30/20 to a new file and validate vs the original
    python preprocess.py --item 30 --user 20 \\
        --output data/reviews_30_20_repro.pkl \\
        --reference data/reviews_30_20.pkl

    # make a 30/10 variant (more users)
    python preprocess.py --item 30 --user 10

    # make a 30/5 variant (even more users)
    python preprocess.py --item 30 --user 5
"""

from __future__ import annotations

import argparse
import logging
import os
import re

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("preprocess")


_HAS_LETTER = re.compile(r"[a-zA-Z]")
_SENT_SPLIT = re.compile(r" *[\.\?!][\'\"\)\]]* *")


# ---------------------------------------------------------------------------
# Filter primitives — each function mirrors a step in the original notebook.
# ---------------------------------------------------------------------------

def clean_review(df: pd.DataFrame) -> pd.DataFrame:
    """verified_purchase=True + drop_duplicates + must contain an English letter."""
    n0 = len(df)
    df = df[df["verified_purchase"] == True].copy()
    df = df.drop_duplicates()
    df = df[df["review_text"].apply(lambda t: bool(_HAS_LETTER.search(str(t))))]
    log.info("clean_review: %d → %d rows", n0, len(df))
    return df


def filter_like_ratio(
    df: pd.DataFrame, lower: float, upper: float
) -> pd.DataFrame:
    """Drop users whose mean(like) ∉ [lower, upper].

    Original notebook drops users with ratio < lower (strict) and ratio >
    upper (strict). Equivalent to keeping ratio ∈ [lower, upper].
    """
    if "like" not in df.columns:
        df = df.copy()
        df["like"] = df["rating"] >= 4

    n_rows0 = len(df)
    n_users0 = df["user_id"].nunique()

    user_like_mean = df.groupby("user_id")["like"].mean()
    keep = user_like_mean[
        (user_like_mean >= lower) & (user_like_mean <= upper)
    ].index
    df = df[df["user_id"].isin(keep)].copy()

    log.info(
        "filter_like_ratio [%.2f, %.2f]: kept %d/%d users, %d/%d rows",
        lower, upper, len(keep), n_users0, len(df), n_rows0,
    )
    return df


def split_review_text(text: object, min_words: int = 3) -> list[str]:
    out: list[str] = []
    for line in str(text).splitlines():
        if line:
            out.extend(_SENT_SPLIT.split(line))
    return [s for s in out if len(s.split()) > min_words]


def drop_empty_review_sentences(df: pd.DataFrame) -> pd.DataFrame:
    """Adds split_review_text column, drops rows with no usable sentences."""
    df = df.copy()
    df["split_review_text"] = df["review_text"].apply(split_review_text)
    n0 = len(df)
    df = df[df["split_review_text"].apply(len) > 0].copy()
    log.info("drop_empty_review_sentences: %d → %d rows", n0, len(df))
    return df


def iter_filter_review_num(
    df: pd.DataFrame,
    item_threshold: int,
    user_threshold: int,
    max_count: int,
) -> pd.DataFrame:
    """Exact mirror of the original notebook's iter_filter_review_num.

    Outer while loop: at most `max_count` iterations OR until both
    item_min ≥ item_threshold and user_min ≥ user_threshold.

    Each iteration: drop items <item_threshold, then drop users <user_threshold.
    """
    c = 0
    item_min, user_min = 0, 0
    while c < max_count and (
        item_min < item_threshold or user_min < user_threshold
    ):
        item_counts = df["asin"].value_counts()
        df = df[df["asin"].isin(item_counts[item_counts >= item_threshold].index)]

        user_counts = df["user_id"].value_counts()
        df = df[df["user_id"].isin(user_counts[user_counts >= user_threshold].index)]

        item_counts = df["asin"].value_counts()
        user_counts = df["user_id"].value_counts()
        item_min = int(item_counts.min()) if len(item_counts) else 0
        user_min = int(user_counts.min()) if len(user_counts) else 0
        c += 1

    log.info(
        "iter_filter [item≥%d user≥%d, max_count=%d]: ran %d iter(s) → "
        "rows=%d items=%d users=%d (actual item_min=%d user_min=%d)",
        item_threshold, user_threshold, max_count, c,
        len(df), df["asin"].nunique(), df["user_id"].nunique(),
        item_min, user_min,
    )
    return df


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def filter_reviews(
    df: pd.DataFrame,
    item_threshold: int,
    user_threshold: int,
    like_lower: float,
    like_upper: float,
    ref_users: set | None = None,
) -> pd.DataFrame:
    """Run the full pipeline. If ref_users given, log per-step survival count."""

    def survival(label: str) -> None:
        if ref_users is None:
            return
        cur = set(df["user_id"].astype(str).unique())
        n = len(ref_users & cur)
        log.info(
            "  ref-survival [%s]: %d/%d users (%.1f%%)",
            label, n, len(ref_users), 100.0 * n / max(1, len(ref_users)),
        )

    log.info(
        "Raw: rows=%d items=%d users=%d",
        len(df), df["asin"].nunique(), df["user_id"].nunique(),
    )
    survival("raw")

    df = clean_review(df)
    survival("after clean_review")

    df["like"] = df["rating"] >= 4
    df = filter_like_ratio(df, like_lower, like_upper)
    survival("after like_ratio (1st)")

    df = iter_filter_review_num(df, item_threshold, user_threshold, max_count=1)
    survival("after iter_filter MAX_COUNT=1")

    df = drop_empty_review_sentences(df)
    survival("after drop_empty_sentences")

    df = iter_filter_review_num(df, item_threshold, user_threshold, max_count=10)
    survival("after iter_filter MAX_COUNT=10")

    df = filter_like_ratio(df, like_lower, like_upper)
    survival("after like_ratio (2nd)")

    df = df.astype({"asin": str, "user_id": str}).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Validation: compare output to a reference pkl
# ---------------------------------------------------------------------------

def validate_against(out_df: pd.DataFrame, ref_path: str) -> None:
    ref = pd.read_pickle(ref_path)

    out_users = set(out_df["user_id"].astype(str).unique())
    out_items = set(out_df["asin"].astype(str).unique())
    ref_users = set(ref["user_id"].astype(str).unique())
    ref_items = set(ref["asin"].astype(str).unique())

    out_pairs = set(zip(out_df["user_id"].astype(str), out_df["asin"].astype(str)))
    ref_pairs = set(zip(ref["user_id"].astype(str), ref["asin"].astype(str)))

    log.info("=" * 60)
    log.info("VALIDATION vs %s", ref_path)
    log.info("=" * 60)
    log.info(
        "rows:    out=%d  ref=%d  diff=%+d",
        len(out_df), len(ref), len(out_df) - len(ref),
    )
    log.info(
        "users:   out=%d  ref=%d  overlap=%d (%.1f%% of ref)",
        len(out_users), len(ref_users), len(out_users & ref_users),
        100.0 * len(out_users & ref_users) / max(1, len(ref_users)),
    )
    log.info(
        "items:   out=%d  ref=%d  overlap=%d (%.1f%% of ref)",
        len(out_items), len(ref_items), len(out_items & ref_items),
        100.0 * len(out_items & ref_items) / max(1, len(ref_items)),
    )
    log.info(
        "pairs:   out=%d  ref=%d  overlap=%d (%.1f%% of ref)",
        len(out_pairs), len(ref_pairs), len(out_pairs & ref_pairs),
        100.0 * len(out_pairs & ref_pairs) / max(1, len(ref_pairs)),
    )
    if out_pairs == ref_pairs:
        log.info("✓ EXACT MATCH on (user_id, asin) pairs")
    elif out_pairs >= ref_pairs:
        log.info("✓ output is a SUPERSET of reference (%d extra pairs)",
                 len(out_pairs - ref_pairs))
    elif out_pairs <= ref_pairs:
        log.info("⚠ output is a SUBSET of reference — missing %d ref pairs",
                 len(ref_pairs - out_pairs))
    else:
        log.info(
            "⚠ partial overlap: out_only=%d  ref_only=%d  shared=%d",
            len(out_pairs - ref_pairs),
            len(ref_pairs - out_pairs),
            len(out_pairs & ref_pairs),
        )


# ---------------------------------------------------------------------------
# I/O safety
# ---------------------------------------------------------------------------

def _safe_save(df: pd.DataFrame, out: str, force: bool) -> None:
    """Refuse to overwrite an existing pkl unless --force is given.

    A previous version of this script silently clobbered
    data/reviews_30_20.pkl when run at thresholds 30/20. Never again.
    """
    if os.path.exists(out) and not force:
        raise FileExistsError(
            f"Refusing to overwrite existing file: {out}\n"
            f"  → pass --force to overwrite, or use --output <other_path>"
        )
    df.to_pickle(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", default="data/Books.pkl",
                   help="Raw review pkl. Required columns: user_id, asin, rating, "
                        "review_text, verified_purchase. (default: data/Books.pkl)")
    p.add_argument("--output", default=None,
                   help="Output pkl. Default: data/reviews_{item}_{user}.pkl")
    p.add_argument("--item", "--item-threshold", type=int, default=30,
                   dest="item_threshold",
                   help="Min reviews per item (default 30 — matches original 30/20).")
    p.add_argument("--user", "--user-threshold", type=int, default=10,
                   dest="user_threshold",
                   help="Min reviews per user (default 10 — relaxed from original 20).")
    p.add_argument("--like-lower", type=float, default=0.3,
                   help="Min mean(like) per user (drop pure-haters). Default 0.3.")
    p.add_argument("--like-upper", type=float, default=0.9,
                   help="Max mean(like) per user (drop pure-lovers). Default 0.9.")
    p.add_argument("--reference", default=None,
                   help="Reference pkl (e.g. reviews_30_20.pkl). If given, logs "
                        "per-step survival of reference users AND validates output "
                        "(user_id, asin) pairs against this reference at the end.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite output file if it already exists.")
    args = p.parse_args()

    log.info("Loading raw from %s", args.input)
    df = pd.read_pickle(args.input)

    ref_users = None
    if args.reference:
        ref = pd.read_pickle(args.reference)
        ref_users = set(ref["user_id"].astype(str).unique())
        log.info(
            "Tracking %d reference users from %s",
            len(ref_users), args.reference,
        )

    out_df = filter_reviews(
        df,
        item_threshold=args.item_threshold,
        user_threshold=args.user_threshold,
        like_lower=args.like_lower,
        like_upper=args.like_upper,
        ref_users=ref_users,
    )

    out_path = args.output or (
        f"data/reviews_{args.item_threshold}_{args.user_threshold}.pkl"
    )
    _safe_save(out_df, out_path, args.force)
    log.info(
        "✓ saved → %s | rows=%d items=%d users=%d like_ratio=%.3f",
        out_path, len(out_df), out_df["asin"].nunique(),
        out_df["user_id"].nunique(),
        out_df["like"].mean() if len(out_df) else float("nan"),
    )

    # KG coverage check (only meaningful if items overlap with the KG)
    kg_path = "data/df_edges_item_aspect1.csv"
    if os.path.exists(kg_path) and len(out_df):
        kg = pd.read_csv(kg_path)
        kg_items = set(kg["node_1"].astype(str).unique())
        out_items = set(out_df["asin"].astype(str).unique())
        covered = out_items & kg_items
        log.info(
            "KG coverage: %d/%d output items have aspect edges (%.1f%%)",
            len(covered), len(out_items),
            100.0 * len(covered) / max(1, len(out_items)),
        )

    if args.reference:
        validate_against(out_df, args.reference)


if __name__ == "__main__":
    main()
