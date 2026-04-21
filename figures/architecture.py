"""
Draw Figure 1: RA-GARK architecture in a clean horizontal left-to-right
swim-lane layout. Local view (collaborative) and Global view (KG semantic)
each flow through one module block into a merged outputs box, then both
feed the per-side fusion gate that produces the final score.

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

# ── Lane Y centres ──────────────────────────────────────────────────────
LANE_LOCAL_Y  = 4.40
LANE_GLOBAL_Y = 2.00
FUSION_Y      = 3.20


# ── Primitives ──────────────────────────────────────────────────────────
def draw_box(ax, x, y, w, h, title, *, color, subtitle=None, bold=True,
             title_size=10, sub_size=8):
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
        ax.text(cx, cy + h * 0.18, title,
                ha="center", va="center",
                fontsize=title_size, weight="bold" if bold else "normal")
        ax.text(cx, cy - h * 0.22, subtitle,
                ha="center", va="center",
                fontsize=sub_size, color="#444444", style="italic")


def draw_arrow(ax, x1, y1, x2, y2, *, dashed=False, lw=1.1,
               color=None, shrink=0.0):
    style = "--" if dashed else "-"
    c = color or ("#888888" if dashed else EDGE_COLOR)
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", linestyle=style,
        color=c, linewidth=lw, mutation_scale=12,
    )
    ax.add_patch(arr)


def draw_star(ax, x, y, caption):
    """Red ★ with caption directly above it."""
    ax.text(x, y, "★", ha="center", va="center",
            fontsize=15, color=COLOR_STAR, weight="bold")
    if caption:
        ax.text(x, y + 0.42, caption,
                ha="center", va="center",
                fontsize=8.2, color=COLOR_STAR,
                style="italic", weight="bold")


def lane_band(ax, y_centre, label, color, *, y_span=1.35, x0=2.0, x1=9.2):
    h = y_span
    ax.add_patch(FancyBboxPatch(
        (x0, y_centre - h / 2), x1 - x0, h,
        boxstyle="round,pad=0.0,rounding_size=0.10",
        linewidth=0, facecolor=color, alpha=0.32,
    ))
    ax.text(x0 + 0.12, y_centre + h / 2 - 0.18, label,
            ha="left", va="top",
            fontsize=9.5, color="#333333", weight="bold", style="italic")


# ── Main figure ─────────────────────────────────────────────────────────
def draw():
    fig, ax = plt.subplots(figsize=(15.5, 5.8))
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 6.0)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title (centered, concise)
    ax.text(7.75, 5.70,
            "RA-GARK Architecture  —  Dual-View + Softmax Rationale + Local-Biased Fusion",
            ha="center", va="center",
            fontsize=12, weight="bold", color="#1F3A68")

    # ── Swim lanes (light backgrounds) ─────────────────────────────────
    lane_band(ax, LANE_LOCAL_Y,  "LOCAL VIEW  (collaborative signal)",
              COLOR_LOCAL,  x0=2.0, x1=9.2)
    lane_band(ax, LANE_GLOBAL_Y, "GLOBAL VIEW  (KG semantic signal)",
              COLOR_GLOBAL, x0=2.0, x1=9.2)

    # ── INPUT ──────────────────────────────────────────────────────────
    draw_box(ax, 0.3, 2.75, 1.45, 0.9, "INPUT",
             color=COLOR_INPUT, subtitle="user u,  item i",
             title_size=11, sub_size=9)

    # ── LOCAL LANE ─────────────────────────────────────────────────────
    # Box 1: user/item local embs
    draw_box(ax, 2.2, LANE_LOCAL_Y - 0.38, 2.0, 0.8,
             "user / item local embs",
             color=COLOR_LOCAL, title_size=9.5, bold=False)
    # Box 2: LightGCN
    draw_box(ax, 4.4, LANE_LOCAL_Y - 0.38, 2.0, 0.8,
             "LightGCN propagation",
             color=COLOR_LOCAL,
             subtitle="D⁻¹ᐟ² A D⁻¹ᐟ² x,  K = 2",
             title_size=9.5, sub_size=8)
    # Box 3: merged outputs
    draw_box(ax, 6.6, LANE_LOCAL_Y - 0.38, 2.4, 0.8,
             "Local outputs",
             color="white", subtitle="u_loc,  i_loc",
             title_size=10, sub_size=9)

    # ── GLOBAL LANE ────────────────────────────────────────────────────
    # Box 1: item_kg_aspects
    draw_box(ax, 2.2, LANE_GLOBAL_Y - 0.38, 2.0, 0.8,
             "item_kg_aspects",
             color=COLOR_GLOBAL, subtitle="[N_i, A = 4, d]",
             title_size=9.5, sub_size=8)
    draw_star(ax, 3.2, LANE_GLOBAL_Y + 0.80, "KG SVD init")

    # Box 2: Rationale masking
    draw_box(ax, 4.4, LANE_GLOBAL_Y - 0.38, 2.0, 0.8,
             "Rationale Masking",
             color=COLOR_GLOBAL,
             subtitle="softmax(MLP(·) / τ),  τ = 0.5",
             title_size=9.5, sub_size=8)
    draw_star(ax, 5.4, LANE_GLOBAL_Y + 0.80, "Softmax + τ")

    # Box 3: merged outputs
    draw_box(ax, 6.6, LANE_GLOBAL_Y - 0.38, 2.4, 0.8,
             "Global outputs",
             color="white",
             subtitle="u_glo (user emb),  i_glo (Σ w·aspect)",
             title_size=10, sub_size=8)

    # ── FUSION GATES (two stacked boxes; one logical module) ───────────
    # Upper: user gate
    draw_box(ax, 9.6, LANE_LOCAL_Y - 0.38, 2.5, 0.8,
             "user fusion gate",
             color=COLOR_FUSION,
             subtitle="α_u = σ(MLP + 5)   →   u_final",
             title_size=9.5, sub_size=8)
    # Lower: item gate
    draw_box(ax, 9.6, LANE_GLOBAL_Y - 0.38, 2.5, 0.8,
             "item fusion gate",
             color=COLOR_FUSION,
             subtitle="α_i = σ(MLP + 5)   →   i_final",
             title_size=9.5, sub_size=8)
    # One shared ★ between the two boxes
    draw_star(ax, 10.85, FUSION_Y, "Local-biased init")

    # ── SCORE ──────────────────────────────────────────────────────────
    draw_box(ax, 12.4, FUSION_Y - 0.40, 2.4, 0.8,
             "score",
             color=COLOR_SCORE,
             subtitle="u_final · i_final",
             title_size=11, sub_size=9)

    # ── LOSSES ─────────────────────────────────────────────────────────
    # Only L_BPR is drawn as a box (it is the main training signal on
    # the score). L_aCL and L_uCL are shown inline between the two
    # lanes as short vertical dashed "CL connectors" — keeping the
    # bottom strip clean.
    draw_box(ax, 11.6, 0.30, 3.6, 0.65,
             "L_BPR   ( +  0.005 · [L_aCL + L_uCL] )",
             color=COLOR_LOSS,
             subtitle="BPR ranking loss on score  +  small CL regulariser",
             title_size=10, sub_size=8)

    # ── SOLID ARROWS (data flow) ───────────────────────────────────────
    # INPUT → lanes
    draw_arrow(ax, 1.75, 3.35, 2.20, LANE_LOCAL_Y - 0.05)
    draw_arrow(ax, 1.75, 3.10, 2.20, LANE_GLOBAL_Y + 0.05)

    # LOCAL lane: embs → LightGCN → outputs
    draw_arrow(ax, 4.20, LANE_LOCAL_Y, 4.40, LANE_LOCAL_Y)
    draw_arrow(ax, 6.40, LANE_LOCAL_Y, 6.60, LANE_LOCAL_Y)

    # GLOBAL lane: aspects → Rationale → outputs
    draw_arrow(ax, 4.20, LANE_GLOBAL_Y, 4.40, LANE_GLOBAL_Y)
    draw_arrow(ax, 6.40, LANE_GLOBAL_Y, 6.60, LANE_GLOBAL_Y)

    # Local outputs → user fusion (straight) + item fusion (diagonal down)
    draw_arrow(ax, 9.00, LANE_LOCAL_Y,  9.60, LANE_LOCAL_Y)
    draw_arrow(ax, 9.00, LANE_LOCAL_Y - 0.20, 9.60, LANE_GLOBAL_Y + 0.20,
               lw=0.9)
    # Global outputs → user fusion (diagonal up) + item fusion (straight)
    draw_arrow(ax, 9.00, LANE_GLOBAL_Y + 0.20, 9.60, LANE_LOCAL_Y - 0.20,
               lw=0.9)
    draw_arrow(ax, 9.00, LANE_GLOBAL_Y, 9.60, LANE_GLOBAL_Y)

    # Fusion gates → score
    draw_arrow(ax, 12.10, LANE_LOCAL_Y,  12.40, FUSION_Y + 0.15)
    draw_arrow(ax, 12.10, LANE_GLOBAL_Y, 12.40, FUSION_Y - 0.15)

    # Score → L_BPR (bottom)
    draw_arrow(ax, 13.40, FUSION_Y - 0.40, 13.40, 0.95)

    # ── INLINE CL CONNECTORS (vertical dashed between the two lanes) ──
    # Each CL loss shown as a short labelled dashed double-arrow
    # between the lanes, at the x of its source pair.
    cl_y_top, cl_y_bot = LANE_LOCAL_Y - 0.40, LANE_GLOBAL_Y + 0.40
    cl_y_mid = (cl_y_top + cl_y_bot) / 2

    # Connector x positions chosen to sit between the ★ novelty markers
    # (★ are at x ≈ 3.2, 5.4, 10.85) so the CL labels don't collide with
    # the ★ captions.
    for cl_x, cl_label, cl_sub in [
        (4.30, "L_aCL", "i_loc ↔ aspects"),     # between aspects and Rationale
        (6.50, "L_uCL", "u_loc ↔ u_glo"),        # between Rationale and outputs
    ]:
        # short dashed stubs near each lane edge
        draw_arrow(ax, cl_x, cl_y_top, cl_x, cl_y_mid + 0.22,
                   dashed=True, lw=0.95)
        draw_arrow(ax, cl_x, cl_y_bot, cl_x, cl_y_mid - 0.22,
                   dashed=True, lw=0.95)
        # label pill in the middle of the gap
        ax.text(cl_x, cl_y_mid + 0.06, cl_label,
                ha="center", va="center",
                fontsize=8.8, color="#444",
                bbox=dict(facecolor="white", edgecolor="#888",
                          boxstyle="round,pad=0.24", linewidth=0.7))
        ax.text(cl_x, cl_y_mid - 0.22, cl_sub,
                ha="center", va="center",
                fontsize=7.2, color="#666", style="italic")

    # ── Legend ─────────────────────────────────────────────────────────
    ax.text(0.30, 0.75, "★  novelty",
            fontsize=9.5, color=COLOR_STAR, style="italic", weight="bold")
    ax.text(0.30, 0.45, "dashed = stop-grad CL",
            fontsize=8.5, color="#555555", style="italic")

    # Save
    Path("figures").mkdir(exist_ok=True)
    fig.tight_layout(pad=0.2)
    fig.savefig("figures/architecture.png", dpi=220, bbox_inches="tight")
    fig.savefig("figures/architecture.pdf", bbox_inches="tight")
    print("Saved: figures/architecture.png and figures/architecture.pdf")


if __name__ == "__main__":
    draw()
