"""Ablation: which of relation_typed_aspect_init / user_svd_init is the regressor?

The both-on configuration (rel_typed=True, user_svd=True) already ran:
NDCG@20 = 0.1183 on seed=42, leaving a 0.006 gap vs the raw-KG 0.124
ceiling. This script runs the two single-flag variants on the same seed
and config so we can attribute the gap.

  A — rel_typed=True,  user_svd=False    # is user-side SVD the culprit?
  B — rel_typed=False, user_svd=True     # is relation-typed item init the culprit?

Both run winner config × seed=42 (configs have collapsed under canonical
KG — winner ≈ scalar_gate ≈ no_cl — so cycling through the other configs
is wasted compute).

Run from repo root:

    python -m relation_init.ablation
    # or
    python relation_init/ablation.py
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import logging

import torch

from relation_init.train import train_ragark, _make_winner_cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# Reference numbers from prior runs, printed in the summary so the
# user doesn't have to scroll back to compare.
REFERENCES = [
    ("raw-KG winner (memory ceiling)        ", 0.124,   None),
    ("canonical legacy adapter (5-seed mean)", 0.1128,  0.0045),
    ("rel_typed=T, user_svd=T  (1-seed)     ", 0.1183,  None),
]


def _format_ref(label: str, ndcg: float, std: float | None) -> str:
    if std is None:
        return f"  {label}  NDCG {ndcg:.4f}"
    return f"  {label}  NDCG {ndcg:.4f} ± {std:.4f}"


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    seed = 42
    runs = [
        ("A_rel_only",  {"relation_typed_aspect_init": True,  "user_svd_init": False}),
        ("B_user_only", {"relation_typed_aspect_init": False, "user_svd_init": True}),
    ]

    results: dict[str, dict | None] = {}
    for name, overrides in runs:
        cfg = _make_winner_cfg(seed)
        for k, v in overrides.items():
            setattr(cfg, k, v)
        cfg.model_save_path = f"best_relinit_ablation_{name}_seed{seed}.pth"

        log.info("\n%s", "=" * 90)
        log.info(
            "ABLATION %s | seed=%d | rel_typed=%s user_svd=%s",
            name, seed,
            cfg.relation_typed_aspect_init, cfg.user_svd_init,
        )
        log.info("%s", "=" * 90)

        try:
            test_res = train_ragark(cfg, device)
        except Exception as e:
            log.exception("%s FAILED: %s", name, e)
            test_res = None
        results[name] = test_res

    # ── Summary ──
    log.info("\n%s", "=" * 100)
    log.info("ABLATION SUMMARY  (seed=%d, winner config)", seed)
    log.info("%s", "=" * 100)
    log.info("Reference:")
    for label, ndcg, std in REFERENCES:
        log.info(_format_ref(label, ndcg, std))
    log.info("")

    log.info("This ablation:")
    log.info("%-15s | %-15s | %8s | %8s | %8s | %8s",
             "Run", "Flags", "NDCG", "Recall", "HR", "MAP")
    log.info("-" * 100)
    label_map = {
        "A_rel_only":  "rel=T user=F",
        "B_user_only": "rel=F user=T",
    }
    for name, _ in runs:
        r = results[name]
        if r is None:
            log.info("%-15s | %-15s | FAILED", name, label_map.get(name, ""))
        else:
            log.info(
                "%-15s | %-15s | %8.4f | %8.4f | %8.4f | %8.4f",
                name, label_map.get(name, ""),
                r["NDCG"], r["Recall"], r["HR"], r["MAP"],
            )
    log.info("=" * 100)

    # ── Quick verdict (printed alongside the table) ──
    nd_a = results["A_rel_only"]["NDCG"] if results["A_rel_only"] else None
    nd_b = results["B_user_only"]["NDCG"] if results["B_user_only"] else None
    nd_both = 0.1183
    if nd_a is not None and nd_b is not None:
        log.info("")
        log.info("Quick read:")
        log.info("  A (rel=T,user=F) − both    = %+.4f", nd_a - nd_both)
        log.info("  B (rel=F,user=T) − both    = %+.4f", nd_b - nd_both)
        if nd_a > 0.122 or nd_b > 0.122:
            log.info("  → at least one variant beats raw-KG 0.124 within noise.")
        elif max(nd_a, nd_b) > nd_both + 0.001:
            log.info("  → one flag carries most of the value; the other is "
                     "neutral-or-negative.")
        else:
            log.info("  → neither flag alone beats both-on; init-only ceiling "
                     "appears to be 0.118.")
