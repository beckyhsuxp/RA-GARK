# Repository Guidelines

## Project Structure & Module Organization

- `Code/` contains the RA-GARK Python implementation. Core files include `model.py`, `data.py`, `train_ragark.py`, `run_ablations.py`, `evaluate.py`, and `config.py`.
- `Code/data/` stores local datasets and KG inputs used by the training scripts.
- `Code/baselines/` contains baseline implementations or adapters.
- `Code/relation_init/` is an archived canonical-KG initialization experiment. Treat it as reproducibility material unless explicitly revisiting that branch.
- `Document/notes/` contains research notes, slide sources, architecture notes, and writing outlines.
- `Document/thesis/` contains the LaTeX thesis, bibliography, figures, and generated PDFs.
- `submit/` contains journal submission artifacts.

## Build, Test, and Development Commands

Run Python commands from `Code/` so relative paths such as `data/reviews_30_20.pkl` resolve correctly:

```bash
cd Code
python train_ragark.py
python run_ablations.py --mode minimal
python run_ablations.py --mode paper --reuse
python case_study.py
```

Build thesis PDFs from `Document/thesis/`:

```bash
cd Document/thesis
latexmk -xelatex -interaction=nonstopmode main.tex
latexmk -xelatex -interaction=nonstopmode main_zh_preview.tex
```

## Coding Style & Naming Conventions

Python code uses 4-space indentation, type hints where helpful, and module-level functions for experiment steps. Keep configuration changes in `Config` or ablation presets instead of scattering constants through scripts. Use descriptive snake_case names for functions, variables, CSV outputs, and experiment tags. Avoid broad refactors when updating paper numbers or a single experiment path.

LaTeX sections live in `Document/thesis/Sections/`. Keep English and Chinese counterparts synchronized, using `_chinese.tex` suffixes for Chinese files.

For thesis-writing changes, use this workflow: revise the English source first, align the Chinese version next, compile the relevant PDF, then commit and push the source and generated PDF together. If LaTeX reports errors but still produces a PDF, commit and push that PDF only when the failure is caused by the known `BiauKaiTC` font problem; for other LaTeX errors, preserve the last good PDF unless the user explicitly asks otherwise.

## Testing Guidelines

There is no formal unit-test framework configured. Validate code changes with the smallest relevant run first, usually:

```bash
cd Code
python run_ablations.py --mode minimal --reuse
```

For LaTeX changes, rebuild the affected PDF and check compile warnings for missing references, missing citations, and overfull boxes. For result-table edits, verify the source CSVs in `Code/` or `Code/runs/` before changing thesis text.

## Commit & Pull Request Guidelines

Recent commits use short, scoped messages such as `paper(jim): ...`, `submit: ...`, and `ref: ...`. Follow that pattern: start with the area, then a concise imperative summary.

Pull requests should describe the changed experiment, paper section, or artifact; list commands run; mention any metrics changed; and include screenshots or PDF page references for layout-sensitive thesis or submission updates.

## Agent-Specific Instructions

Do not overwrite generated best-run ledgers or checkpoints unless the task explicitly requires rerunning experiments. Preserve archived negative results in `Code/relation_init/`. When editing thesis content, prefer updating notes and source `.tex` files over generated auxiliary files.
