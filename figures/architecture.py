"""
Draw Figure 1: RA-GARK architecture with the three ★-novelties
highlighted. Saves PNG and PDF for paper use.

Run:
    python figures/architecture.py
Outputs:
    figures/architecture.png
    figures/architecture.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


# ── Style constants ─────────────────────────────────────────────────────
COLOR_LOCAL   = "#E6F0FF"
COLOR_GLOBAL  = "#FFF0E0"
COLOR_FUSION  = "#E8E8F8"
COLOR_LOSS    = "#F8F8E0"
COLOR_STAR    = "#D83A3A"
EDGE_COLOR    = "#333333"
FONT_LABEL    = dict(fontsize=10, ha="center", va="center")
FONT_STAR     = dict(fontsize=16, color=COLOR_STAR, ha="center", va="center", weight="bold")
FONT_CAPTION  = dict(fontsize=9, color=COLOR_STAR, ha="center", va="center", style="italic")


def _box(ax, x, y, w, h, text, color, subtext=None, *, bold=False):
    """Rounded box with centred text."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        linewidth=1.1, edgecolor=EDGE_COLOR, facecolor=color,
    )
    ax.add_patch(box)
    tfont = FONT_LABEL.copy()
    if bold:
        tfont["weight"] = "bold"
    if subtext is None:
        ax.text(x + w / 2, y + h / 2, text, **tfont)
    else:
        ax.text(x + w / 2, y + h * 0.63, text, **tfont)
        sub = tfont.copy()
        sub.update(fontsize=8, color="#444444", style="italic", weight="normal")
        ax.text(x + w / 2, y + h * 0.28, subtext, **sub)


def _star(ax, x, y, caption=None):
    ax.text(x, y, "★", **FONT_STAR)
    if caption is not None:
        ax.text(x, y - 0.30, caption, **FONT_CAPTION)


def _arrow(ax, x1, y1, x2, y2, *, dashed=False, label=None):
    style = "-" if not dashed else "--"
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", linestyle=style,
        color=EDGE_COLOR, linewidth=1.1, mutation_scale=12,
    )
    ax.add_patch(arr)
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.05, label, fontsize=8,
                ha="center", va="bottom", style="italic", color="#555555")


def draw():
    fig, ax = plt.subplots(figsize=(12, 7.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.text(6.0, 7.2,
            "RA-GARK: Dual-View Recommendation with Softmax Rationale Attention and Local-Biased Fusion",
            fontsize=12, ha="center", va="center", weight="bold")

    # Input
    _box(ax, 5.0, 6.25, 2.0, 0.55, "INPUT (u, i)", "#F0F0F0", bold=True)

    # ── Local view column (left) ────────────────────────────────────────
    _box(ax, 0.6, 5.0, 3.6, 0.55, "user_local_emb, item_local_emb", COLOR_LOCAL)
    _box(ax, 0.6, 4.1, 3.6, 0.65, "LightGCN propagation",
         COLOR_LOCAL, subtext="D⁻¹ᐟ² A D⁻¹ᐟ² x,  K=2,  layer-mean")
    _box(ax, 1.2, 3.05, 1.35, 0.55, "u_loc", COLOR_LOCAL, bold=True)
    _box(ax, 2.85, 3.05, 1.35, 0.55, "i_loc", COLOR_LOCAL, bold=True)

    # ── Global view column (right) ──────────────────────────────────────
    _box(ax, 7.8, 5.0, 3.6, 0.55, "item_kg_aspects  [Ni, A, d]", COLOR_GLOBAL, bold=True)
    _star(ax, 7.6, 5.27, caption="KG SVD init")

    _box(ax, 7.8, 4.1, 3.6, 0.65, "KG Rationale Masking",
         COLOR_GLOBAL,
         subtext="weights = softmax(MLP([u_glo; i_aspects]))")
    _star(ax, 7.6, 4.44, caption="Softmax\nnormalise")

    _box(ax, 7.8, 3.05, 1.6, 0.55, "u_glo", COLOR_GLOBAL)
    _box(ax, 9.8, 3.05, 1.6, 0.55, "i_glo = Σ w·aspect", COLOR_GLOBAL)

    # ── Fusion gate row ─────────────────────────────────────────────────
    _box(ax, 1.4, 1.85, 3.4, 0.65, "user_fusion_gate",
         COLOR_FUSION, subtext="α_u = σ(MLP + b=+5)  →  u_final")
    _box(ax, 7.2, 1.85, 3.4, 0.65, "item_fusion_gate",
         COLOR_FUSION, subtext="α_i = σ(MLP + b=+5)  →  i_final")
    _star(ax, 4.9, 2.25, caption="Local-biased init\n(α≈0.993 at start)")
    _star(ax, 7.0, 2.25, caption=None)

    # Score and losses
    _box(ax, 4.9, 0.85, 2.2, 0.6, "score = u_final · i_final", "#FFFFFF", bold=True)
    _box(ax, 0.4, 0.05, 2.2, 0.5, "L_BPR", COLOR_LOSS, bold=True)
    _box(ax, 4.9, 0.05, 2.2, 0.5, "L_aCL   (aspect CL, stop-grad)", COLOR_LOSS)
    _box(ax, 9.4, 0.05, 2.2, 0.5, "L_uCL   (user CL, stop-grad)", COLOR_LOSS)

    # ── Arrows: input → views ───────────────────────────────────────────
    _arrow(ax, 5.6, 6.25, 2.5, 5.60)                         # in → local emb
    _arrow(ax, 6.4, 6.25, 9.6, 5.60)                         # in → global emb

    # Local chain
    _arrow(ax, 2.4, 5.00, 2.4, 4.78)                         # emb → lightgcn
    _arrow(ax, 2.0, 4.08, 1.85, 3.65)                        # lightgcn → u_loc
    _arrow(ax, 2.8, 4.08, 3.5, 3.65)                         # lightgcn → i_loc

    # Global chain
    _arrow(ax, 9.6, 5.00, 9.6, 4.78)                         # aspects → rationale
    _arrow(ax, 8.4, 4.08, 8.6, 3.65)                         # rationale → u_glo (shown as input below)
    _arrow(ax, 9.8, 4.08, 10.6, 3.65)                        # rationale → i_glo
    # user_global_emb feeds rationale — draw a small box above "u_glo"
    _arrow(ax, 8.6, 3.05, 8.6, 4.10, dashed=True, label="condition")

    # Into fusion
    _arrow(ax, 1.85, 3.05, 2.2, 2.52)                        # u_loc → user_fusion
    _arrow(ax, 8.6, 3.05, 7.8, 2.52)                         # u_glo → user_fusion
    _arrow(ax, 3.5, 3.05, 4.4, 2.52, dashed=False)           # (cross) — not needed; remove
    _arrow(ax, 3.5, 3.05, 8.3, 2.52)                         # i_loc → item_fusion
    _arrow(ax, 10.6, 3.05, 9.3, 2.52)                        # i_glo → item_fusion

    # Into score
    _arrow(ax, 3.1, 1.85, 5.5, 1.48)
    _arrow(ax, 8.9, 1.85, 6.5, 1.48)

    # Score → losses
    _arrow(ax, 5.5, 0.85, 1.5, 0.56)                         # score → L_BPR

    # CL arrows (dashed — stop-grad)
    _arrow(ax, 3.5, 3.05, 6.0, 0.56, dashed=True)            # i_loc → aCL
    _arrow(ax, 10.6, 3.05, 6.0, 0.56, dashed=True)           # aspects → aCL  (stop-grad target)
    _arrow(ax, 1.85, 3.05, 10.4, 0.56, dashed=True)          # u_loc → uCL
    _arrow(ax, 8.6, 3.05, 10.4, 0.56, dashed=True)           # u_glo → uCL  (stop-grad target)

    # ── Legend for stars ────────────────────────────────────────────────
    ax.text(0.1, 6.8, "★  = novelty", **FONT_CAPTION)

    Path("figures").mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig("figures/architecture.png", dpi=200, bbox_inches="tight")
    fig.savefig("figures/architecture.pdf", bbox_inches="tight")
    print("Saved: figures/architecture.png and figures/architecture.pdf")


if __name__ == "__main__":
    draw()
