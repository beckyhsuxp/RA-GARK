"""
Run a battery of ablations over RA-GARK's architectural flags and dump
test metrics to CSV for easy comparison.

Three preset groups — pick one with --mode:

    minimal  (4 presets, ~8 min)
        winner + each of the 3 ★ novelties reverted.
        Smallest set that still populates every row of the paper's
        Section-4 novelty claims.

    paper    (7 presets, ~14 min)  [default]
        minimal + old_full (pre-Fix baseline) + the CL ablations
        (winner_no_acl, winner_no_ucl) + lightgcn_only (no-KG floor).
        This is the ablation table that goes into the paper.

    full     (10 presets, ~20 min)
        paper + winner_no_rat + no_global_view.

Run:
    python run_ablations.py                 # paper (default)
    python run_ablations.py --mode minimal  # fastest
    python run_ablations.py --mode full     # everything

Output: ablation_results.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import time

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
    "use_rationale", "use_svd_init",
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
    tag_bits.append(f"fb{cfg.fusion_init_bias:.0f}")
    cfg.model_save_path = f"best_ragark_{'_'.join(tag_bits)}.pth"
    return cfg


# All presets are defined once; the --mode flag chooses which subset to run.
ALL_PRESETS = {
    "winner":             {},                                                  # ≈ 0.1231
    "winner_sigmoid_rat": {"rationale_style": "mlp_sigmoid"},                  # proves softmax ★
    "winner_no_svd":      {"use_svd_init": False},                             # proves KG SVD ★
    "winner_fb0":         {"fusion_init_bias": 0.0},                           # proves fusion bias ★
    "winner_no_acl":      {"use_acl": False},                                  # supporting
    "winner_no_ucl":      {"use_ucl": False},                                  # supporting
    "winner_no_rat":      {"use_rationale": False},                            # rationale off
    "old_full":           {"rationale_style": "mlp_sigmoid",                   # ≈ 0.1064
                           "fusion_init_bias": 0.0},
    "no_global_view":     {"use_global_view": False},                          # ≈ 0.1218
    "lightgcn_only":      {"use_global_view": False, "use_rationale": False,   # ≈ 0.1179
                           "use_acl": False, "use_ucl": False},
}

MODES = {
    "minimal": [
        "winner",
        "winner_sigmoid_rat",
        "winner_no_svd",
        "winner_fb0",
    ],
    "paper": [
        "winner",
        "winner_sigmoid_rat",
        "winner_no_svd",
        "winner_fb0",
        "winner_no_acl",
        "winner_no_ucl",
        "old_full",
        "lightgcn_only",
    ],
    "full": list(ALL_PRESETS.keys()),
}


def run_presets(mode: str) -> list[tuple[str, dict]]:
    names = MODES[mode]
    return [(n, ALL_PRESETS[n]) for n in names]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=list(MODES.keys()), default="paper",
        help="Which preset group to run (default: paper).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s  |  Mode: %s  (%d presets)",
             device, args.mode, len(MODES[args.mode]))

    results = []
    presets = run_presets(args.mode)
    t_total = time.perf_counter()

    for i, (name, overrides) in enumerate(presets, 1):
        log.info("=" * 70)
        log.info("[%d/%d] Running preset: %s  (overrides=%s)",
                 i, len(presets), name, overrides)
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

    out = f"ablation_results_{args.mode}.csv"
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
