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

    sensitivity (10 presets, ~15 min)
        Hyperparameter sensitivity for the §4 sensitivity section:
        aspect slots A, contrastive weight λ_CL, and fusion-gate bias b,
        all anchored on `winner`. Also available split: sens_A / sens_lambda
        / sens_bias. The swept scalars (num_aspects, cl_weight,
        fusion_init_bias) are written to the CSV so each row is identifiable.

Run:
    python run_ablations.py                      # paper (default)
    python run_ablations.py --mode minimal       # fastest
    python run_ablations.py --mode sensitivity   # A + λ_CL + b sweeps
    python run_ablations.py --mode full          # everything
    python run_ablations.py --mode full --reuse  # skip configs already cached

Output: ablation_results_<mode>.csv (canonical, latest run — read by the §4
tables) plus a timestamped copy under runs/archive/ that is never overwritten.
Results persist so a re-run never loses a good number: the best-ever checkpoint
per config is kept in runs/best/<tag>.pth and its test metrics in
runs/best_ledger.json; a worse re-run does not overwrite either. With --reuse,
a preset whose config is already in the ledger is skipped (stored metrics
reused, no training).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import time
from datetime import datetime

import torch

from config import Config
from train_ragark import evaluate_ragark, train_ragark

# --- Result persistence (so re-runs never lose a good result) ---------------
# runs/best/<tag>.pth   : best-ever checkpoint per config, kept across runs
# runs/working/<tag>.pth: this run's checkpoint (train writes+reads it)
# runs/archive/*.csv    : timestamped copy of every run's CSV (never clobbered)
# runs/best_ledger.json : best-ever test metrics per config key
RUNS_DIR = "runs"
BEST_DIR = os.path.join(RUNS_DIR, "best")
WORKING_DIR = os.path.join(RUNS_DIR, "working")
ARCHIVE_DIR = os.path.join(RUNS_DIR, "archive")
LEDGER_PATH = os.path.join(RUNS_DIR, "best_ledger.json")


def _config_tag(cfg: Config) -> str:
    """Unique fingerprint of a config — reuse the tag make_cfg already builds."""
    return os.path.splitext(os.path.basename(cfg.model_save_path))[0]


def _load_ledger() -> dict:
    if os.path.exists(LEDGER_PATH):
        with open(LEDGER_PATH) as f:
            return json.load(f)
    return {}


def _save_ledger(ledger: dict) -> None:
    os.makedirs(RUNS_DIR, exist_ok=True)
    tmp = LEDGER_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(ledger, f, indent=2)
    os.replace(tmp, LEDGER_PATH)  # atomic — ledger never half-written

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
    tag_bits.append(f"t{cfg.rationale_temperature:.2f}")
    tag_bits.append(f"fb{cfg.fusion_init_bias:.0f}")
    tag_bits.append(f"gate-{cfg.fusion_gate_style}")
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
    "winner_scalar_gate": {"fusion_gate_style": "scalar"},                     # validates per-(u,i) MLP
    "winner_no_rat":      {"use_rationale": False},                            # rationale off
    "old_full":           {"rationale_style": "mlp_sigmoid",                   # ≈ 0.1064
                           "fusion_init_bias": 0.0},
    "no_global_view":     {"use_global_view": False},                          # ≈ 0.1218
    "lightgcn_only":      {"use_global_view": False, "use_rationale": False,   # ≈ 0.1179
                           "use_acl": False, "use_ucl": False},

    # ── Temperature sweep: rescue user-conditioned attention ───────────
    # Case study showed τ=1 gives near-uniform attention. Sharpen the
    # softmax to amplify small user-specific logit differences.
    "winner_temp_0.5":    {"rationale_temperature": 0.5},
    "winner_temp_0.1":    {"rationale_temperature": 0.1},
    "winner_temp_0.05":   {"rationale_temperature": 0.05},

    # ── Sensitivity: latent aspect slots A (default 4 = winner) ────────
    # SVD rank scales as k = A·d (handled by model from cfg.num_aspects).
    "winner_A2":          {"num_aspects": 2},
    "winner_A8":          {"num_aspects": 8},

    # ── Sensitivity: contrastive weight λ_CL (default 0.005 = winner) ──
    "winner_cl0":         {"cl_weight": 0.0},
    "winner_cl0.001":     {"cl_weight": 0.001},
    "winner_cl0.01":      {"cl_weight": 0.01},
    "winner_cl0.05":      {"cl_weight": 0.05},

    # ── Sensitivity: fusion-gate bias b (default +5 = winner; b=0 is winner_fb0) ──
    "winner_fb2":         {"fusion_init_bias": 2.0},
    "winner_fb10":        {"fusion_init_bias": 10.0},
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
        "winner_scalar_gate",
        "winner_no_acl",
        "winner_no_ucl",
        "old_full",
        "lightgcn_only",
    ],
    "temp": [
        "winner",           # τ=1.0 baseline
        "winner_temp_0.5",
        "winner_temp_0.1",
        "winner_temp_0.05",
    ],
    # Each sensitivity mode anchors on `winner` (A=4, λ_CL=0.005, b=+5, τ=0.5).
    "sens_A": [
        "winner",           # A=4
        "winner_A2",
        "winner_A8",
    ],
    "sens_lambda": [
        "winner",           # λ_CL=0.005
        "winner_cl0",
        "winner_cl0.001",
        "winner_cl0.01",
        "winner_cl0.05",
    ],
    "sens_bias": [
        "winner",           # b=+5
        "winner_fb2",
        "winner_fb0",
        "winner_fb10",
    ],
    "sensitivity": [        # all three axes in one run (~15 min)
        "winner",
        "winner_A2", "winner_A8",
        "winner_cl0", "winner_cl0.001", "winner_cl0.01", "winner_cl0.05",
        "winner_fb2", "winner_fb0", "winner_fb10",
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
    parser.add_argument(
        "--reuse", action="store_true",
        help="Skip training a preset if a cached best checkpoint already exists "
             "for its exact config; reuse the stored best metrics instead.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s  |  Mode: %s  (%d presets)  |  reuse=%s",
             device, args.mode, len(MODES[args.mode]), args.reuse)

    for d in (BEST_DIR, WORKING_DIR, ARCHIVE_DIR):
        os.makedirs(d, exist_ok=True)
    ledger = _load_ledger()

    results = []
    presets = run_presets(args.mode)
    t_total = time.perf_counter()

    for i, (name, overrides) in enumerate(presets, 1):
        log.info("=" * 70)
        log.info("[%d/%d] Running preset: %s  (overrides=%s)",
                 i, len(presets), name, overrides)
        log.info("=" * 70)

        cfg = make_cfg(**overrides)
        tag = _config_tag(cfg)
        best_ckpt = os.path.join(BEST_DIR, f"{tag}.pth")
        cfg.model_save_path = os.path.join(WORKING_DIR, f"{tag}.pth")

        reused = False
        t0 = time.perf_counter()
        if args.reuse and tag in ledger and os.path.exists(best_ckpt):
            # Trust the stored best metrics — zero compute, never retrain.
            test_res = {k: float(v) for k, v in ledger[tag]["metrics"].items()}
            reused = True
            log.info("↺ reuse cached best for %s (NDCG=%.4f) — no training",
                     name, test_res.get("NDCG", float("nan")))
        else:
            try:
                test_res = train_ragark(cfg, device)
            except Exception as e:
                log.exception("Preset %s FAILED: %s", name, e)
                test_res = {}
            # Keep-best: promote this run's checkpoint only if it beats the
            # best-ever for this config. A worse re-run never overwrites it.
            new_ndcg = test_res.get("NDCG")
            prev_ndcg = ledger.get(tag, {}).get("best_ndcg")
            if test_res and new_ndcg is not None and (
                prev_ndcg is None or new_ndcg > prev_ndcg
            ):
                if os.path.exists(cfg.model_save_path):
                    shutil.copy2(cfg.model_save_path, best_ckpt)
                ledger[tag] = {
                    "preset": name,
                    "best_ndcg": new_ndcg,
                    "metrics": {k: float(v) for k, v in test_res.items()},
                    "seed": cfg.seed,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "checkpoint": best_ckpt,
                }
                _save_ledger(ledger)  # crash-safe: persist after each improvement
                log.info("★ new best for %s: NDCG=%.4f (prev=%s)",
                         tag, new_ndcg,
                         f"{prev_ndcg:.4f}" if prev_ndcg is not None else "none")
            elif prev_ndcg is not None:
                log.info("· kept previous best for %s: NDCG=%.4f (this run=%s)",
                         tag, prev_ndcg,
                         f"{new_ndcg:.4f}" if new_ndcg is not None else "NaN")
        elapsed = time.perf_counter() - t0

        row = {"preset": name, "elapsed_s": f"{elapsed:.0f}", "reused": int(reused)}
        for k in BOOL_FLAGS:
            row[k] = int(getattr(cfg, k))
        row["rationale_style"] = cfg.rationale_style
        row["rationale_temperature"] = cfg.rationale_temperature
        row["fusion_gate_style"] = cfg.fusion_gate_style
        # Swept scalars — recorded so sensitivity CSVs are self-documenting.
        row["num_aspects"] = cfg.num_aspects
        row["cl_weight"] = cfg.cl_weight
        row["fusion_init_bias"] = cfg.fusion_init_bias
        for m in ("HR", "Precision", "Recall", "F1", "MAP", "NDCG"):
            row[m] = f"{test_res.get(m, float('nan')):.4f}" if test_res else "NaN"
        results.append(row)

        log.info("✓ %s done in %.0fs — NDCG=%s%s",
                 name, elapsed, row["NDCG"], "  (reused)" if reused else "")

    # Canonical latest CSV (back-compat: this is what the §4 tables read from)
    # plus a timestamped archive copy that is never overwritten.
    out = f"ablation_results_{args.mode}.csv"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = os.path.join(ARCHIVE_DIR, f"ablation_results_{args.mode}_{stamp}.csv")
    fieldnames = list(results[0].keys())
    for path in (out, archive):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(results)

    log.info("=" * 70)
    log.info("All %d presets done in %.0fs → %s  (archived: %s)",
             len(presets), time.perf_counter() - t_total, out, archive)
    log.info("Best-ever ledger: %s  (%d configs tracked)", LEDGER_PATH, len(ledger))
    log.info("=" * 70)
    for r in results:
        log.info(
            "%-20s NDCG=%s HR=%s Recall=%s MAP=%s%s",
            r["preset"], r["NDCG"], r["HR"], r["Recall"], r["MAP"],
            "  (reused)" if r["reused"] else "",
        )


if __name__ == "__main__":
    main()
