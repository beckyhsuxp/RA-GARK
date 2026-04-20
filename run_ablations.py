"""
Run a battery of ablations over the 5 architectural flags in RA-GARK and
dump test metrics to CSV for easy comparison.

Presets covered (all use cl_weight=0.005, epochs=80, seed=42):

    full             all flags ON (current RA-GARK)
    no_rationale     uniform mean over aspects
    no_svd           xavier init instead of KG SVD
    no_kg_lr         single global lr
    no_acl           drop aspect-level CL
    no_ucl           drop user cross-view CL

Baseline config is tested once, then each "no_X" variant flips one flag off.
Since previous runs showed `no_rationale` > full, a second pass ablates
each component *from the no_rationale baseline* too — the winning config —
to see what's actually carrying the lift.

Run:
    python run_ablations.py
Output:
    ablation_results.csv
"""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import asdict

import torch

from config import Config
from train_ragark import train_ragark

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ablate")

BOOL_FLAGS = (
    "use_rationale", "use_svd_init", "use_kg_lr",
    "use_acl", "use_ucl", "use_global_view",
)


def make_cfg(**overrides) -> Config:
    cfg = Config()
    cfg.cl_weight = 0.005
    cfg.epochs = 80
    for k, v in overrides.items():
        setattr(cfg, k, v)
    tag_bits = [f"{k.replace('use_', '')}{int(getattr(cfg, k))}" for k in BOOL_FLAGS]
    tag_bits.append(f"style-{cfg.rationale_style}")
    cfg.model_save_path = f"best_ragark_{'_'.join(tag_bits)}.pth"
    return cfg


def run_presets():
    # Presets from the previous round (1-10) kept for reproducibility.
    # New presets (11+): use_global_view ablation and rationale-rescue variants.
    # The previous winner was no_rat_no_ucl (NDCG 0.1204).
    presets = [
        # ── sanity: previous results ────────────────────────────────
        ("full",                       {}),
        ("no_rationale",               {"use_rationale": False}),
        ("no_rat_no_ucl",              {"use_rationale": False, "use_ucl": False}),

        # ── NEW: pure LightGCN / global-view ablations ──────────────
        ("no_global_view",             {"use_global_view": False}),
        ("lightgcn_only",              {
            "use_global_view": False, "use_rationale": False,
            "use_acl": False, "use_ucl": False,
        }),

        # ── NEW: rescue rationale with softmax / dot-product heads ──
        ("rat_softmax",                {"rationale_style": "mlp_softmax"}),
        ("rat_dot",                    {"rationale_style": "dot_softmax"}),

        # Pair the new rationale heads with the winning config
        # (no_ucl, everything else full) to see if they beat 0.1204
        ("rat_softmax_no_ucl",         {"rationale_style": "mlp_softmax", "use_ucl": False}),
        ("rat_dot_no_ucl",             {"rationale_style": "dot_softmax", "use_ucl": False}),
    ]
    return presets


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    results = []
    presets = run_presets()
    t_total = time.perf_counter()

    for i, (name, overrides) in enumerate(presets, 1):
        log.info("=" * 70)
        log.info("[%d/%d] Running preset: %s  (overrides=%s)", i, len(presets), name, overrides)
        log.info("=" * 70)

        cfg = make_cfg(**overrides)
        t0 = time.perf_counter()
        try:
            test_res = train_ragark(cfg, device)
        except Exception as e:
            log.exception("Preset %s FAILED: %s", name, e)
            test_res = {}
        elapsed = time.perf_counter() - t0

        row = {"preset": name, "elapsed_s": f"{elapsed:.0f}"}
        for k in BOOL_FLAGS:
            row[k] = int(getattr(cfg, k))
        row["rationale_style"] = cfg.rationale_style
        for m in ("HR", "Precision", "Recall", "F1", "MAP", "NDCG"):
            row[m] = f"{test_res.get(m, float('nan')):.4f}" if test_res else "NaN"
        results.append(row)

        log.info("✓ %s done in %.0fs — NDCG=%s", name, elapsed, row["NDCG"])

    out = "ablation_results.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    log.info("=" * 70)
    log.info("All %d presets done in %.0fs → %s",
             len(presets), time.perf_counter() - t_total, out)
    log.info("=" * 70)
    for r in results:
        log.info(
            "%-20s NDCG=%s HR=%s Recall=%s MAP=%s",
            r["preset"], r["NDCG"], r["HR"], r["Recall"], r["MAP"],
        )


if __name__ == "__main__":
    main()
