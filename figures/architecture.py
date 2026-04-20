"""
Draw Figure 1: RA-GARK architecture as a horizontal left-to-right pipeline.
Two parallel lanes (LOCAL and GLOBAL) flow into a fusion gate that
produces the final score. The three ★-novelties are highlighted inline.

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


# ── Palette ─────────────────────────────────────────────────────────────
COLOR_INPUT   = "#E8E8E8"
COLOR_LOCAL   = "#DCE8FB"
COLOR_GLOBAL  = "#FCE6CE"
COLOR_FUSION  = "#E4E0F5"
COLOR_SCORE   = "#FFFFFF"
COLOR_LOSS    = "#F5F1C8"
COLOR_STAR    = "#D83A3A"
EDGE_COLOR    = "#2A2A2A"

LANE_LOCAL_Y  = 4.5       # centre Y of the LOCAL lane
LANE_GLOBAL_Y = 2.0       # centre Y of the GLOBAL lane
FUSION_Y      = 3.25      # centre Y of the fusion gate block


# ── Primitives ──────────────────────────────────────────────────────────
def draw_box(ax, x, y, w, h, title, *, color, subtitle=None, bold=True,
             title_size=10, sub_size=8):
    """Rounded rectangle with centred title (and optional subtitle)."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        linewidth=1.0, edgecolor=EDGE_COLOR, facecolor=color,
    )
    ax.add_patch(box)
    cx, cy = x + w / 2, y + h / 2
    if subtitle is None:
        ax.text(cx, cy, title,
                ha="center", va="center",
                fontsize=title_size, weight="bold" if bold else "normal")
    else:
        ax.text(cx, cy + h * 0.17, title,
                ha="center", va="center",
                fontsize=title_size, weight="bold" if bold else "normal")
        ax.text(cx, cy - h * 0.22, subtitle,
                ha="center", va="center",
                fontsize=sub_size, color="#444444", style="italic")
    return (cx, cy, x, y, x + w, y + h)


def draw_arrow(ax, x1, y1, x2, y2, *, dashed=False, label=None, label_dy=0.12,
               lw=1.1):
    style = "--" if dashed else "-"
    color = "#777777" if dashed else EDGE_COLOR
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", linestyle=style,
        color=color, linewidth=lw, mutation_scale=11,
    )
    ax.add_patch(arr)
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + label_dy, label,
                ha="center", va="bottom",
                fontsize=7.5, color="#555555", style="italic")


def draw_star(ax, x, y, caption, *, dx=0.0, dy=0.55):
    """Red ★ with italic caption above it."""
    ax.text(x, y, "★", ha="center", va="center",
            fontsize=14, color=COLOR_STAR, weight="bold")
    if caption:
        ax.text(x + dx, y + dy, caption,
                ha="center", va="center",
                fontsize=7.8, color=COLOR_STAR, style="italic", weight="bold")


def lane_band(ax, y_centre, label, color, *, y_span=1.2, x0=0.3, x1=15.8):
    """Faint background band to signal the swim-lane."""
    h = y_span
    ax.add_patch(FancyBboxPatch(
        (x0, y_centre - h / 2), x1 - x0, h,
        boxstyle="round,pad=0.0,rounding_size=0.10",
        linewidth=0, facecolor=color, alpha=0.35,
    ))
    ax.text(x0 + 0.15, y_centre + h / 2 - 0.22, label,
            ha="left", va="top",
            fontsize=9.5, color="#333333", weight="bold", style="italic")


# ── Main figure ─────────────────────────────────────────────────────────
def draw():
    fig, ax = plt.subplots(figsize=(16, 6.0))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6.2)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title (right-aligned so it doesn't collide with the input box)
    ax.text(8.0, 5.95,
            "RA-GARK: Dual-View Recommendation with Softmax Aspect Saliency and Local-Biased Fusion",
            ha="center", va="center",
            fontsize=11.5, weight="bold", color="#1F3A68")

    # ── Swim lanes ─────────────────────────────────────────────────────
    lane_band(ax, LANE_LOCAL_Y,  "LOCAL VIEW  (collaborative signal)",
              COLOR_LOCAL,  y_span=1.3, x0=1.9, x1=10.0)
    lane_band(ax, LANE_GLOBAL_Y, "GLOBAL VIEW  (KG semantic signal)",
              COLOR_GLOBAL, y_span=1.3, x0=1.9, x1=10.0)

    # ── INPUT ─────────────────────────────────────────────────────────
    draw_box(ax, 0.3, 2.9, 1.5, 0.8,
             "INPUT", color=COLOR_INPUT, subtitle="user u,  item i",
             title_size=11, sub_size=9)

    # ── LOCAL lane: embeddings → LightGCN → u_loc / i_loc ──────────────
    draw_box(ax, 2.1, LANE_LOCAL_Y - 0.3, 2.1, 0.7,
             "user / item\nlocal embs",
             color=COLOR_LOCAL, title_size=9, bold=False)
    draw_box(ax, 4.5, LANE_LOCAL_Y - 0.35, 2.1, 0.8,
             "LightGCN",
             color=COLOR_LOCAL, subtitle="D⁻¹ᐟ² A D⁻¹ᐟ² x,  K=2",
             title_size=10, sub_size=8)
    draw_box(ax, 6.9, LANE_LOCAL_Y - 0.3, 1.3, 0.7,
             "u_loc", color="white", title_size=10)
    draw_box(ax, 8.4, LANE_LOCAL_Y - 0.3, 1.3, 0.7,
             "i_loc", color="white", title_size=10)

    # ── GLOBAL lane: item_kg_aspects / u_glo → Rationale → i_glo ───────
    draw_box(ax, 2.1, LANE_GLOBAL_Y - 0.3, 2.1, 0.7,
             "item_kg_aspects",
             color=COLOR_GLOBAL, subtitle="[N, A=4, d]",
             title_size=9, sub_size=8)
    draw_star(ax, 2.1 + 2.1 - 0.2, LANE_GLOBAL_Y + 0.6,
              "KG SVD init", dx=0.0, dy=0.32)

    draw_box(ax, 4.5, LANE_GLOBAL_Y - 0.35, 2.1, 0.8,
             "Rationale Masking",
             color=COLOR_GLOBAL,
             subtitle="softmax(MLP(·) / τ),  τ = 0.5",
             title_size=10, sub_size=8)
    draw_star(ax, 4.5 + 2.1 - 0.2, LANE_GLOBAL_Y + 0.65,
              "Softmax + τ", dx=0.0, dy=0.32)

    draw_box(ax, 6.9, LANE_GLOBAL_Y - 0.3, 1.3, 0.7,
             "u_glo", color="white", title_size=10)
    draw_box(ax, 8.4, LANE_GLOBAL_Y - 0.3, 1.3, 0.7,
             "i_glo", color="white", title_size=10)

    # ── FUSION GATES (one per side) ────────────────────────────────────
    draw_box(ax, 10.3, LANE_LOCAL_Y - 0.35, 2.2, 0.85,
             "user fusion gate",
             color=COLOR_FUSION,
             subtitle="α_u = σ(MLP + 5)   →   u_final",
             title_size=9.5, sub_size=8)
    draw_box(ax, 10.3, LANE_GLOBAL_Y - 0.35, 2.2, 0.85,
             "item fusion gate",
             color=COLOR_FUSION,
             subtitle="α_i = σ(MLP + 5)   →   i_final",
             title_size=9.5, sub_size=8)
    draw_star(ax, 10.3 + 2.2 + 0.3, FUSION_Y + 0.05,
              "Local-biased\ninit  bias = +5", dx=0.0, dy=0.48)

    # ── SCORE ──────────────────────────────────────────────────────────
    draw_box(ax, 13.2, FUSION_Y - 0.4, 2.4, 0.8,
             "score",
             color=COLOR_SCORE,
             subtitle="u_final · i_final",
             title_size=11, sub_size=9)

    # ── LOSSES (bottom) ────────────────────────────────────────────────
    draw_box(ax, 1.9, 0.3, 2.3, 0.55,
             "L_aCL", color=COLOR_LOSS, title_size=9, bold=True)
    ax.text(1.9 + 1.15, 0.15,
            "aspect-level CL (stop-grad)",
            ha="center", va="center",
            fontsize=7.3, color="#555555", style="italic")

    draw_box(ax, 4.6, 0.3, 2.3, 0.55,
             "L_uCL", color=COLOR_LOSS, title_size=9, bold=True)
    ax.text(4.6 + 1.15, 0.15,
            "user cross-view CL (stop-grad)",
            ha="center", va="center",
            fontsize=7.3, color="#555555", style="italic")

    draw_box(ax, 13.0, 0.3, 2.6, 0.55,
             "L_BPR",
             color=COLOR_LOSS, subtitle="L_total = L_BPR + 0.005·(L_aCL+L_uCL)",
             title_size=9, sub_size=7.5)

    # ── Arrows (left → right) ──────────────────────────────────────────
    # INPUT → lanes
    draw_arrow(ax, 1.8, 3.3, 2.1, LANE_LOCAL_Y - 0.02)
    draw_arrow(ax, 1.8, 3.3, 2.1, LANE_GLOBAL_Y + 0.02)

    # LOCAL lane forward
    draw_arrow(ax, 4.2, LANE_LOCAL_Y, 4.5, LANE_LOCAL_Y)
    draw_arrow(ax, 6.6, LANE_LOCAL_Y, 6.9, LANE_LOCAL_Y)
    draw_arrow(ax, 8.2, LANE_LOCAL_Y, 8.4, LANE_LOCAL_Y)

    # GLOBAL lane forward
    draw_arrow(ax, 4.2, LANE_GLOBAL_Y, 4.5, LANE_GLOBAL_Y)
    draw_arrow(ax, 6.6, LANE_GLOBAL_Y, 6.9, LANE_GLOBAL_Y)
    draw_arrow(ax, 8.2, LANE_GLOBAL_Y, 8.4, LANE_GLOBAL_Y)

    # u_glo feeds back as conditioning into Rationale Masking (dashed)
    draw_arrow(ax, 7.55, LANE_GLOBAL_Y + 0.4, 5.55, LANE_GLOBAL_Y + 0.05,
               dashed=True, label="condition")

    # Lanes → Fusion gates (u_loc+u_glo → user gate, i_loc+i_glo → item gate)
    draw_arrow(ax, 7.55, LANE_LOCAL_Y,  10.3, LANE_LOCAL_Y)                 # u_loc
    draw_arrow(ax, 7.55, LANE_GLOBAL_Y, 10.3, LANE_GLOBAL_Y)                # u_glo — into user gate via cross
    draw_arrow(ax, 9.05, LANE_LOCAL_Y,  10.3, LANE_LOCAL_Y,  lw=0.8)        # (overlap no-op)
    draw_arrow(ax, 9.05, LANE_GLOBAL_Y, 10.3, LANE_GLOBAL_Y, lw=0.8)
    # Cross flow: u_glo also needs to reach user fusion gate (same lane)
    # i_loc needs to reach item fusion gate (same lane) — already covered above
    draw_arrow(ax, 7.55, LANE_GLOBAL_Y + 0.1, 10.3, LANE_LOCAL_Y - 0.2,
               dashed=False, lw=0.9)
    draw_arrow(ax, 7.55, LANE_LOCAL_Y - 0.1, 10.3, LANE_GLOBAL_Y + 0.2,
               dashed=False, lw=0.9)

    # Fusion → score (one arrow from user gate, one from item gate)
    draw_arrow(ax, 12.5, LANE_LOCAL_Y - 0.05,  13.2, FUSION_Y + 0.2)
    draw_arrow(ax, 12.5, LANE_GLOBAL_Y + 0.05, 13.2, FUSION_Y - 0.2)

    # Score → L_BPR
    draw_arrow(ax, 14.4, FUSION_Y - 0.4, 14.4, 0.85)

    # CL losses (dashed, stop-grad) from the lanes
    # aCL pulls from i_loc ↔ aspects
    draw_arrow(ax, 9.05, LANE_LOCAL_Y - 0.3, 3.0, 0.85, dashed=True)
    draw_arrow(ax, 3.15, LANE_GLOBAL_Y - 0.3, 3.05, 0.85, dashed=True)
    # uCL pulls from u_loc ↔ u_glo
    draw_arrow(ax, 7.55, LANE_LOCAL_Y - 0.3, 5.75, 0.85, dashed=True)
    draw_arrow(ax, 7.55, LANE_GLOBAL_Y - 0.3, 5.75, 0.85, dashed=True)

    # ── Legend ─────────────────────────────────────────────────────────
    ax.text(0.3, 0.55, "★  novelty",
            fontsize=9, color=COLOR_STAR, style="italic", weight="bold")
    ax.text(0.3, 0.25, "dashed = stop-grad / CL path",
            fontsize=8, color="#555555", style="italic")

    # Save
    Path("figures").mkdir(exist_ok=True)
    fig.tight_layout(pad=0.2)
    fig.savefig("figures/architecture.png", dpi=220, bbox_inches="tight")
    fig.savefig("figures/architecture.pdf", bbox_inches="tight")
    print("Saved: figures/architecture.png and figures/architecture.pdf")


if __name__ == "__main__":
    draw()
