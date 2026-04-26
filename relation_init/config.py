"""Config for the relation-typed-init experiment.

Inherits root Config so every existing knob stays available; adds the two
flags that turn on the kg_init.py SVD machinery. Defaults are flipped ON
since the only reason to be in this package is to run the experiment —
to fall back to root behaviour without leaving the package, set both
flags to False.
"""
from __future__ import annotations

from dataclasses import dataclass

from config import Config  # absolute import, resolves to repo-root config.py


@dataclass
class RelationInitConfig(Config):
    # When True: item_kg_aspects[i, a, :] = SVD on the relation-group-a
    # subset of item-aspect edges. Each of the 4 aspects becomes a
    # canonical relation type (HAS_PROPERTY / DEPICTS / IS_A / OTHER-bucket)
    # instead of an anonymous SVD component. Reads kg_canonical.csv.
    relation_typed_aspect_init: bool = True

    # When True: user_global_emb is SVD-initialised from the user-side
    # canonical KG. Parallel of the existing build_kg_aspect_init trick
    # extended to user side; doesn't touch model architecture.
    user_svd_init: bool = True
