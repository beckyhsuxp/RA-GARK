"""
RAKG-LMR v4 — No InfoNCE (cl_weight = 0), Optuna reg_weight kept.
Tests whether the InfoNCE contrastive loss contributes at all.

Compare against v3 (cl=0.000258, reg=0.973, NDCG=0.1155).

Run:
    python train_v4.py
"""

import logging
import torch
from config import Config
from train_v1 import train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

if __name__ == "__main__":
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.getLogger(__name__).info("Device: %s", _device)

    cfg = Config()
    cfg.cl_weight       = 0.0     # no InfoNCE
    cfg.reg_weight      = 0.973   # keep Optuna reg_weight
    cfg.model_save_path = "best_model_v4.pth"

    train(cfg, _device)
