"""Run the main benchmark models and export Top-20/Top-10 metrics to CSV.

This script is meant for the thesis tables. It evaluates the four KG-aware
baselines plus the LightGCN floor and the full RA-GARK model, then writes a
single machine-readable CSV with both @20 and @10 metrics.

Run:
    cd Code
    python run_main_benchmark.py
"""

from __future__ import annotations

import csv
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import torch

from baselines.train_kgat import train_kgat
from baselines.train_kgcl import train_kgcl
from baselines.train_kgrec import train_kgrec
from baselines.train_mcclk import train_mcclk
from config import Config
from train_ragark import train_ragark

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main-benchmark")

OUT_CSV = "main_benchmark_results.csv"
ARCHIVE_DIR = Path("runs") / "archive"


def _run_baseline(name: str, fn, cfg: Config, device: torch.device) -> dict:
    log.info("Running %s", name)
    t0 = time.perf_counter()
    metrics = fn(cfg, device)
    elapsed = time.perf_counter() - t0
    row = {"model": name, "elapsed_s": f"{elapsed:.0f}"}
    for k, v in metrics.items():
        try:
            row[k] = f"{float(v):.4f}"
        except (TypeError, ValueError):
            row[k] = v
    return row


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    rows: list[dict] = []

    baseline_specs = [
        ("MCCLK", train_mcclk, "best_model_mcclk.pth", {}),
        ("KGCL", train_kgcl, "best_model_kgcl.pth", {}),
        ("KGAT", train_kgat, "best_model_kgat.pth", {}),
        ("KGRec", train_kgrec, "best_model_kgrec.pth", {}),
    ]

    for name, fn, ckpt, overrides in baseline_specs:
        local_cfg = Config()
        local_cfg.epochs = 80
        local_cfg.model_save_path = ckpt
        for k, v in overrides.items():
            setattr(local_cfg, k, v)
        rows.append(_run_baseline(name, fn, local_cfg, device))

    ragark_cfg = Config()
    ragark_cfg.use_rationale = False
    ragark_cfg.use_svd_init = True
    ragark_cfg.use_acl = False
    ragark_cfg.use_ucl = False
    ragark_cfg.use_global_view = False
    ragark_cfg.model_save_path = "best_model_lightgcn_only.pth"
    rows.append(_run_baseline("LightGCN", train_ragark, ragark_cfg, device))

    full_cfg = Config()
    full_cfg.epochs = 80
    full_cfg.model_save_path = "best_ragark_model.pth"
    rows.append(_run_baseline("RA-GARK", train_ragark, full_cfg, device))

    fieldnames = ["model", "elapsed_s"]
    metric_order = [
        "HR", "Precision", "Recall", "F1", "MAP", "NDCG",
        "HR@10", "Precision@10", "Recall@10", "F1@10", "MAP@10", "NDCG@10",
    ]
    fieldnames.extend(metric_order)

    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = ARCHIVE_DIR / f"main_benchmark_results_{stamp}.csv"
    for path in (Path(OUT_CSV), archive_path):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    log.info("Wrote %s and %s", OUT_CSV, archive_path)


if __name__ == "__main__":
    main()
