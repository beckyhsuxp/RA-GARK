from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")
os.environ.setdefault("HOME", "/private/tmp")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update(
    {
        "font.size": 12.5,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    }
)


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "Code"
THESIS_IMG_DIR = ROOT / "Document" / "thesis" / "img"

ARCH_LINE = "#1f2937"
ARCH_HIGHLIGHT = "#b91c1c"
ARCH_GRID = "#d1d5db"
ARCH_REF = "#9ca3af"
ARCH_TEXT = "#111827"
ARCH_HOTMAP = plt.get_cmap("Blues")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def by_preset(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["preset"]: row for row in rows}


def to_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def style_axes(ax: plt.Axes, ylim: tuple[float, float] = (0.112, 0.126)) -> None:
    ax.set_ylim(*ylim)
    ax.grid(axis="y", color=ARCH_GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(ARCH_REF)
    ax.spines["bottom"].set_color(ARCH_REF)
    ax.tick_params(colors=ARCH_TEXT)
    ax.yaxis.label.set_color(ARCH_TEXT)
    ax.xaxis.label.set_color(ARCH_TEXT)


def plot_sensitivity_2x2(rows: dict[str, dict[str, str]]) -> None:
    panels = [
        (
            "Softmax temperature $\\tau$",
            ["1.0", "0.5", "0.1", "0.05"],
            ["winner_temp_1.0", "winner_temp_0.5", "winner_temp_0.1", "winner_temp_0.05"],
            1,
        ),
        (
            "Aspect slots $A$",
            ["2", "3", "4", "6", "8"],
            ["winner_A2", "winner_A3", "winner", "winner_A6", "winner_A8"],
            2,
        ),
        (
            "Fusion bias $b$",
            ["0", "+2", "+5", "+10"],
            ["winner_fb0", "winner_fb2", "winner", "winner_fb10"],
            2,
        ),
        (
            "Contrastive weight $\\lambda_{\\mathrm{CL}}$",
            ["0", "0.001", "0.005", "0.01", "0.05"],
            ["winner_cl0", "winner_cl0.001", "winner", "winner_cl0.01", "winner_cl0.05"],
            2,
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.2))
    fig.subplots_adjust(wspace=0.24, hspace=0.28, left=0.08, right=0.98, top=0.96, bottom=0.08)
    axes = axes.flatten()
    for idx, (title, labels, presets, highlight_idx) in enumerate(panels):
        ax = axes[idx]
        values = [to_float(rows[preset], "NDCG") for preset in presets]
        xs = list(range(len(labels)))
        ax.plot(xs, values, color=ARCH_LINE, linewidth=1.9, marker="o", markersize=5.0)
        ax.scatter([highlight_idx], [values[highlight_idx]], s=70, color=ARCH_HIGHLIGHT, zorder=3)
        ax.axhline(0.1243, color=ARCH_REF, linestyle="--", linewidth=1.0)
        ax.set_title(title, pad=8)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_ylabel("NDCG@20")
        style_axes(ax)

    THESIS_IMG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(THESIS_IMG_DIR / "sensitivity_2x2.pdf", bbox_inches="tight")
    fig.savefig(THESIS_IMG_DIR / "sensitivity_2x2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_case_heatmap(rows: list[dict[str, str]]) -> None:
    items: list[str] = []
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        asin = row["asin"]
        if asin not in grouped:
            items.append(asin)
            grouped[asin] = []
        grouped[asin].append(row)

    fig, axes = plt.subplots(2, 3, figsize=(13.6, 6.1))
    fig.subplots_adjust(wspace=0.18, hspace=0.44, left=0.05, right=0.96, top=0.95, bottom=0.06)
    axes_flat = axes.flatten()
    cmap = ARCH_HOTMAP
    vmin, vmax = 0.225, 0.287
    mappable = None

    for idx, asin in enumerate(items):
        ax = axes_flat[idx]
        item_rows = grouped[asin]
        matrix = [[float(item_row[f"w{i}"]) for i in range(4)] for item_row in item_rows]
        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        mappable = im
        ax.set_title(f"ASIN {asin}", fontsize=10.5, pad=5)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["w0", "w1", "w2", "w3"])
        ax.set_yticks(range(len(item_rows)))
        ax.set_yticklabels([row["user_idx"] for row in item_rows])
        ax.tick_params(axis="both", labelsize=10.5)

        for y, item_row in enumerate(item_rows):
            for x in range(4):
                value = float(item_row[f"w{x}"])
                color = "white" if value > 0.26 else ARCH_TEXT
                ax.text(x, y, f"{value:.3f}", ha="center", va="center", fontsize=9.5, color=color)

        for spine in ax.spines.values():
            spine.set_visible(False)

    for idx in range(len(items), len(axes_flat)):
        axes_flat[idx].axis("off")

    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=axes_flat.tolist(), shrink=0.92, pad=0.02)
        cbar.set_label("Weight")

    THESIS_IMG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(THESIS_IMG_DIR / "case_study_heatmap.pdf", bbox_inches="tight")
    fig.savefig(THESIS_IMG_DIR / "case_study_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ablation_rows = by_preset(load_rows(CODE_DIR / "ablation_results_full.csv"))
    case_rows = load_rows(CODE_DIR / "case_study.csv")
    plot_sensitivity_2x2(ablation_rows)
    plot_case_heatmap(case_rows)


if __name__ == "__main__":
    main()
