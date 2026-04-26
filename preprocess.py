"""
Re-filter raw Amazon Books reviews with configurable k-core thresholds.

Adapted from the original `preprocess.ipynb` (其他 repo)，stripped down to the
filter pipeline RA-GARK actually consumes (no LDA / negative sampling /
train-test split — those are handled in `train_ragark.py` via
`user_stratified_split`).

Two modes:
  --mode fixed-items  (default)
    Lock the item set to an existing reference pkl (`reviews_30_20.pkl`),
    relax only the user-side threshold. KG (`df_edges_item_aspect1.csv`)
    stays 100% aligned — single-variable diagnostic for "more users, same
    items" experiments.

  --mode standard
    Full k-core re-filter from raw `Books.pkl`. Item set will drift relative
    to 30/20, so any new items lack aspect-KG edges (degenerate SVD init).

Output: `data/reviews_{ITEM}_{USER}.pkl` (standard) or
        `data/reviews_30_20items_user{USER}.pkl` (fixed-items)

Usage:
    python preprocess.py                                  # fixed-items, user≥10
    python preprocess.py --user-threshold 5               # fixed-items, user≥5
    python preprocess.py --mode standard                  # full 30/10 re-filter
"""

from __future__ import annotations

import argparse
import logging
import re

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("preprocess")


# ---------------------------------------------------------------------------
# Cleaning steps (preserved from original to keep semantics aligned with 30/20)
# ---------------------------------------------------------------------------

def _has_letter(text: object) -> bool:
    return bool(re.search(r"[a-zA-Z]", str(text)))


def clean_review(df: pd.DataFrame) -> pd.DataFrame:
    """Verified purchase + drop duplicates + reviews must contain English letters."""
    n0 = len(df)
    df = df[df["verified_purchase"] == True].copy()
    df = df.drop_duplicates()
    df = df[df["review_text"].apply(_has_letter)]
    log.info("clean_review: %d → %d rows (dropped %d)", n0, len(df), n0 - len(df))
    return df


def filter_like_ratio(
    df: pd.DataFrame, like_lower: float, like_upper: float
) -> pd.DataFrame:
    """Drop users whose mean(like) ∉ [like_lower, like_upper]."""
    if "like" not in df.columns:
        df = df.copy()
        df["like"] = df["rating"] >= 4

    user_like_mean = df.groupby("user_id")["like"].mean()
    keep = user_like_mean[
        (user_like_mean >= like_lower) & (user_like_mean <= like_upper)
    ].index

    n_users_before = user_like_mean.shape[0]
    n_rows_before = len(df)
    df = df[df["user_id"].isin(keep)].copy()
    log.info(
        "filter_like_ratio [%.2f, %.2f]: kept %d/%d users, %d/%d rows",
        like_lower, like_upper,
        len(keep), n_users_before,
        len(df), n_rows_before,
    )
    return df


_SENTENCE_SPLIT_RE = re.compile(r" *[\.\?!][\'\"\)\]]* *")


def _split_review(text: object, min_words: int = 3) -> list[str]:
    sentences = []
    for line in str(text).splitlines():
        if line:
            sentences.extend(_SENTENCE_SPLIT_RE.split(line))
    return [s for s in sentences if len(s.split()) > min_words]


def drop_empty_review_sentences(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror原 notebook 的 split_review_text 過濾步驟，砍掉拆完無有效句子的 row."""
    df = df.copy()
    df["split_review_text"] = df["review_text"].apply(_split_review)
    n0 = len(df)
    df = df[df["split_review_text"].apply(len) > 0].copy()
    log.info("drop_empty_review_sentences: %d → %d rows", n0, len(df))
    return df


def iter_kcore(
    df: pd.DataFrame,
    item_min: int,
    user_min: int,
    user_max: int | None = None,
    max_iters: int = 10,
) -> pd.DataFrame:
    """Iteratively enforce min(asin count) ≥ item_min and
    user_min ≤ count(user) ≤ user_max until stable.

    Faster than the original by short-circuiting when len() doesn't change
    (the original always ran a fixed iteration budget).
    """
    for it in range(max_iters):
        n0 = len(df)

        item_counts = df["asin"].value_counts()
        df = df[df["asin"].isin(item_counts[item_counts >= item_min].index)]

        user_counts = df["user_id"].value_counts()
        keep = user_counts[user_counts >= user_min]
        if user_max is not None:
            keep = keep[keep <= user_max]
        df = df[df["user_id"].isin(keep.index)]

        if len(df) == n0:
            log.info("iter_kcore converged after %d iter(s)", it + 1)
            break
    else:
        log.warning("iter_kcore did not converge in %d iterations", max_iters)

    item_counts = df["asin"].value_counts()
    user_counts = df["user_id"].value_counts()
    log.info(
        "iter_kcore result: rows=%d items=%d users=%d "
        "(item_min=%d user_min=%d user_max=%s actual_item_min=%d user_min=%d max=%d)",
        len(df), df["asin"].nunique(), df["user_id"].nunique(),
        item_min, user_min, user_max if user_max is not None else "∞",
        int(item_counts.min()) if len(item_counts) else 0,
        int(user_counts.min()) if len(user_counts) else 0,
        int(user_counts.max()) if len(user_counts) else 0,
    )
    return df


# ---------------------------------------------------------------------------
# Mode runners
# ---------------------------------------------------------------------------

def _safe_save(df: pd.DataFrame, out: str, force: bool) -> None:
    """Refuse to overwrite an existing pkl unless --force is given.

    Defensive: a previous version of this script silently overwrote
    data/reviews_30_20.pkl (the precious reference) when run in standard
    mode at item=30 user=20. Never again.
    """
    import os
    if os.path.exists(out) and not force:
        raise FileExistsError(
            f"Refusing to overwrite existing file: {out}\n"
            f"  → pass --force to overwrite, or use --output <other_path> to save elsewhere"
        )
    df.to_pickle(out)


def run_standard(args: argparse.Namespace) -> None:
    log.info("Mode: standard (full k-core re-filter from raw)")
    log.info("Loading raw reviews from %s", args.input)
    df = pd.read_pickle(args.input)
    log.info(
        "Raw: rows=%d items=%d users=%d",
        len(df), df["asin"].nunique(), df["user_id"].nunique(),
    )

    df = clean_review(df)
    df["like"] = df["rating"] >= 4

    df = filter_like_ratio(df, args.like_lower, args.like_upper)
    df = iter_kcore(df, args.item_threshold, args.user_threshold, args.user_max)
    df = drop_empty_review_sentences(df)
    df = iter_kcore(df, args.item_threshold, args.user_threshold, args.user_max)
    df = filter_like_ratio(df, args.like_lower, args.like_upper)

    df = df.astype({"asin": str, "user_id": str}).reset_index(drop=True)

    out = args.output or f"data/reviews_{args.item_threshold}_{args.user_threshold}_v2.pkl"
    _safe_save(df, out, args.force)
    log.info(
        "✓ saved → %s | rows=%d items=%d users=%d like_ratio=%.3f",
        out, len(df), df["asin"].nunique(), df["user_id"].nunique(),
        df["like"].mean() if len(df) else float("nan"),
    )


def run_fixed_items(args: argparse.Namespace) -> None:
    log.info("Mode: fixed-items (lock item set to reference, relax user only)")
    log.info("Loading reference split from %s", args.reference)
    ref = pd.read_pickle(args.reference)
    fixed_items = set(ref["asin"].astype(str).unique())
    ref_users = set(ref["user_id"].astype(str).unique())
    log.info("Reference: %d items, %d users", len(fixed_items), len(ref_users))

    def _survival(df: pd.DataFrame, label: str) -> None:
        """Log how many of the reference 30/20 users survive at this step."""
        cur = set(df["user_id"].astype(str).unique())
        surv = ref_users & cur
        log.info(
            "  ref-survival [%s]: %d/%d users (%.1f%%)",
            label, len(surv), len(ref_users),
            100.0 * len(surv) / max(1, len(ref_users)),
        )

    log.info("Loading raw reviews from %s", args.input)
    df = pd.read_pickle(args.input)
    log.info(
        "Raw: rows=%d items=%d users=%d",
        len(df), df["asin"].nunique(), df["user_id"].nunique(),
    )
    _survival(df, "raw")

    df = clean_review(df)
    _survival(df, "after clean_review")

    df["like"] = df["rating"] >= 4

    # IMPORTANT: like_ratio runs on the FULL cleaned Books corpus first
    # (matches original preprocess.ipynb ordering). If we restricted to the
    # 1398 reference items first, like_ratio would be computed on tiny
    # per-user subsets (1-2 reviews) and almost everyone collapses to 0/1
    # ratio → gets dropped by [0.3, 0.9] band.
    df = filter_like_ratio(df, args.like_lower, args.like_upper)
    _survival(df, "after like_ratio (full corpus)")

    # NOW lock item set.
    df = df[df["asin"].astype(str).isin(fixed_items)].copy()
    log.info(
        "Restricted to reference items: rows=%d items=%d users=%d",
        len(df), df["asin"].nunique(), df["user_id"].nunique(),
    )
    _survival(df, "after restrict-to-ref-items")

    # item_min=1 → don't drop any reference items mid-iter; only user-side moves.
    df = iter_kcore(df, item_min=1, user_min=args.user_threshold, user_max=args.user_max)
    _survival(df, "after iter_kcore #1 (user≥%d)" % args.user_threshold)

    df = drop_empty_review_sentences(df)
    _survival(df, "after drop_empty_sentences")

    df = iter_kcore(df, item_min=1, user_min=args.user_threshold, user_max=args.user_max)
    _survival(df, "after iter_kcore #2")

    df = filter_like_ratio(df, args.like_lower, args.like_upper)
    _survival(df, "after final like_ratio")

    df = df.astype({"asin": str, "user_id": str}).reset_index(drop=True)

    out = args.output or f"data/reviews_30_20items_user{args.user_threshold}.pkl"
    _safe_save(df, out, args.force)
    log.info(
        "✓ saved → %s | rows=%d items=%d users=%d like_ratio=%.3f",
        out, len(df), df["asin"].nunique(), df["user_id"].nunique(),
        df["like"].mean() if len(df) else float("nan"),
    )

    # Sanity check: KG coverage of final item set
    kg_path = "data/df_edges_item_aspect1.csv"
    try:
        kg = pd.read_csv(kg_path)
        kg_items = set(kg["node_1"].astype(str).unique())
        final_items = set(df["asin"].unique())
        covered = final_items & kg_items
        log.info(
            "KG coverage: %d/%d items have aspect edges (%.1f%%)",
            len(covered), len(final_items),
            100.0 * len(covered) / max(1, len(final_items)),
        )
    except Exception as e:  # pragma: no cover
        log.warning("Could not check KG coverage: %s", e)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--mode", choices=["fixed-items", "standard"], default="fixed-items",
        help="fixed-items: lock to reference item set (KG stays aligned). "
             "standard: full re-filter from raw (item set drifts).",
    )
    p.add_argument("--input", default="data/Books.pkl",
                   help="Raw review pkl. Required columns: user_id, asin, rating, "
                        "review_text, verified_purchase. (default: data/Books.pkl)")
    p.add_argument("--reference", default="data/reviews_30_20.pkl",
                   help="[fixed-items] pkl whose item set is locked.")
    p.add_argument("--output", default=None,
                   help="Output path. Default auto-named.")
    p.add_argument("--item-threshold", type=int, default=30,
                   help="Min reviews per item (standard mode). Default 30.")
    p.add_argument("--user-threshold", type=int, default=10,
                   help="Min reviews per user. Default 10.")
    p.add_argument("--user-max", type=int, default=500,
                   help="Max reviews per user (drop power-reviewers / bots). "
                        "Set 0 to disable. Default 500 (matches config.json).")
    p.add_argument("--like-lower", type=float, default=0.3,
                   help="Min mean(like) per user (drop pure-haters). Default 0.3.")
    p.add_argument("--like-upper", type=float, default=0.9,
                   help="Max mean(like) per user (drop pure-lovers). Default 0.9.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite output file if it already exists.")
    args = p.parse_args()

    if args.user_max == 0:
        args.user_max = None

    if args.mode == "standard":
        run_standard(args)
    else:
        run_fixed_items(args)


if __name__ == "__main__":
    main()
