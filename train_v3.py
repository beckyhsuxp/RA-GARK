"""
RAKG-LMR v3 — Optuna-tuned loss weights.
  cl_weight  = 0.000258  (Optuna best, was 0.01)
  reg_weight = 0.973     (Optuna best, was 0.1)

TEST @ K=20:  HR=0.4950  Recall=0.1908  NDCG=0.1155  MAP=0.0539  (best epoch 25)

Run:
    python train_v3.py
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
    cfg.cl_weight       = 0.000258
    cfg.reg_weight      = 0.973
    cfg.model_save_path = "best_model_v3.pth"

    train(cfg, _device)
