# RA-GARK Code Directory

Run all commands from this directory so relative paths under `data/` resolve
correctly.

## Core Files

- `config.py` - central experiment configuration.
- `data.py`, `kg_loader.py` - interaction and knowledge-graph loading.
- `model.py`, `losses.py`, `evaluate.py`, `utils.py` - model, objective,
  metrics, and shared helpers.
- `train_ragark.py` - main RA-GARK training/evaluation entry point.
- `run_ablations.py` - ablation presets and best-run ledger handling.
- `run_main_benchmark.py` - baseline plus RA-GARK benchmark export.

## Directories

- `data/` - local datasets and KG inputs. This directory is ignored by git.
- `baselines/` - KGAT, KGCL, KGRec, MCCLK baseline implementations.
- `relation_init/` - archived canonical-KG initialization experiment.
- `results/` - latest machine-readable outputs used by tables and figures.
- `runs/` - experiment checkpoints, ledgers, and timestamped archives.

## Common Commands

```bash
python train_ragark.py
python run_ablations.py --mode minimal --reuse
python run_ablations.py --mode paper --reuse
python case_study.py
```
