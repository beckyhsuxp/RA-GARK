# relation_init/ — Canonical-KG Init Experiment (archived)

Tried using the canonical KG (`data/kg_canonical.csv`, produced by
`kg_clean.py`) to drive SVD-based initialisation of `item_kg_aspects`
and `user_global_emb`, hoping to match or beat the raw-KG NDCG ceiling
of **0.1240** on `data/reviews_30_20.pkl`.

**Outcome (2026-04-26): init-only ceiling is 0.118; the remaining
0.006 gap is structural — the fusion gate is neutralised under the
canonical KG.** Paper main path stays on raw KG; this folder is kept
as a reproducible record of the negative result.

## What it does

Two SVD-based inits, both additive — root `model.py` / `data.py` /
`train_ragark.py` are not modified:

1. **`build_relation_typed_aspect_init`**  
   Each of the 4 rows in `item_kg_aspects` is initialised by an SVD
   over a *separate* relation-group of canonical-KG edges:
   - aspect 0 ← `HAS_PROPERTY`
   - aspect 1 ← `DEPICTS`
   - aspect 2 ← `IS_A`
   - aspect 3 ← `SET_IN ∪ OTHER ∪ INFLUENCES ∪ {POS,NEG,NEU}_PREF ∪ INTERESTED_IN`

   Replaces the pooled-SVD output of `data.build_kg_aspect_init`. The
   rationale-attention's per-aspect weights are now interpretable as
   relation-type attention.

2. **`build_user_kg_init`**  
   SVD-init for `user_global_emb`, parallel of the existing item-side
   trick applied to the user-side canonical KG (905 users × 4992
   entities × 41,330 edges across 7 user-relevant relations).

Both flags default `True` in `RelationInitConfig`; flip to `False` to
fall back to the root pipeline.

## Files

| File | Purpose |
|---|---|
| `__init__.py` | empty package marker |
| `config.py` | `RelationInitConfig(Config)` — adds the 2 flags |
| `kg_init.py` | the two SVD-init functions |
| `train.py` | self-contained mirror of root `train_ragark.py` + extra init block |
| `ablation.py` | runs the 2 single-flag variants |

## Results (winner config, seed=42)

| Setup | NDCG@20 | Recall@20 | HR@20 |
|---|---|---|---|
| Raw-KG baseline *(memory ceiling)* | **0.1240** | — | — |
| Canonical legacy adapter *(5-seed mean)* | 0.1128 ± 0.0045 | 0.1816 ± 0.0088 | 0.4785 ± 0.0145 |
| rel_typed=T, user_svd=T *(1-seed)* | **0.1183** | 0.1930 | 0.4906 |
| rel_typed=T, user_svd=F *(ablation A)* | 0.1167 | 0.1933 | 0.4983 |
| rel_typed=F, user_svd=T *(ablation B)* | 0.1162 | 0.1884 | 0.4917 |

Combined inits beat each single-flag variant — they are synergistic.
Neither flag is a regressor.

## Why we walked away

Across all canonical-KG configurations, the **architectural levers
collapse**:

```
winner − scalar_gate  ≈ -0.0002   (raw-KG: +0.0058 / +4.8%, validated)
winner − no_cl        ≈ +0.0007   (raw-KG: +0.0050 / +4.1%, validated)
```

The 0.006 NDCG gap to the raw-KG 0.1240 ceiling **matches exactly**
the fusion gate's normal contribution on raw KG. The per-(user,item)
MLP gate becomes indistinguishable from a single global α — its
ability to differentiate local-view from global-view is lost when
the global view is canonical-KG-derived.

Closing this gap requires a structural modification to the fusion
gate (or its input feature plumbing), which would invalidate the
fusion-gate-validates contribution recorded in
`project_fusion_gate_validated.md`. Trade-off was unfavourable:
gain *real-KG / typed-relation interpretability*, lose 5% NDCG **and**
one validated contribution.

## How to reproduce

```bash
git pull

# both-on (default in this folder; reproduces 0.1183 NDCG)
python -m relation_init.train

# the 2 single-flag ablations (reproduces 0.1167 / 0.1162)
python -m relation_init.ablation
```

Wall-clock: ~10–15 min per run on the training GPU machine.

## Future-work entry point

If revisiting this direction: the architectural surgery to recover
the 0.006 gap under canonical KG would target the fusion gate. Either
its input features need an additional channel that captures the
typed-relation structure of the global view, or the gate itself needs
to be relation-aware. That work is explicitly out of scope for the
current paper (touches a validated contribution); flag it as future
work in the limitations section.
