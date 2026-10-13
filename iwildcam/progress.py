"""Draw the team's progress chart from the experiment reports.

    uv run python -m iwildcam.progress
    uv run python -m iwildcam.progress --dark --out experiments/progress-dark.png

Every experiment has one report, `experiments/NNN-category-name.md`, where NNN is its
PR number (see `experiments/README.md`). This script reads, from each report, the
`This` column of the seed table, the `**Verdict:**` line and the `**Test:**` line, and
draws one chart, `experiments/progress.png`:

* one dot per experiment -- the mean validation macro F1 (present) of its seeds, with a
  thin whisker over the seed range, so a gain smaller than the seed spread shows as such;
* kept experiments in colour and labelled, dropped ones in grey;
* the current baseline as a step line: each kept experiment becomes the new baseline;
* the keep bar as a light band above that line -- how much a new experiment must gain
  to be kept (1.15 x the paired seed noise, estimated from the baseline's seeds);
* each test read (milestones only) as a hollow triangle.

Higher is better. Reports with no seed scores yet are skipped.
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = ROOT / "experiments"
REPORT = re.compile(r"(\d{3,})-(.+)\.md")
NUMBER = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")

# Chart tokens, light and dark. One accent hue for kept work, grey for the rest.
THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "kept": "#2a78d6", "dropped": "#c3c2b7",
              "test": "#eb6834", "band": (42 / 255, 120 / 255, 214 / 255, 0.10)},
    "dark": {"surface": "#1a1a19", "text": "#f0efec", "text2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "kept": "#3987e5", "dropped": "#5f5e5a",
             "test": "#d95926", "band": (57 / 255, 135 / 255, 229 / 255, 0.16)},
}

# Keep rule from experiments/README.md: mean paired gain > 1.15 x sigma_d, with
# sigma_d ~ sqrt(2) x the seed std until paired differences are measured.
KEEP_FACTOR = 1.15


def _number(text: str) -> float | None:
    """The first number in `text`, or None."""
    m = NUMBER.search(text)
    return float(m[0]) if m else None


def parse_report(text: str) -> dict:
    """Read the scores out of one report.

    Args:
        text: the Markdown of the report.

    Returns:
        `seeds` (the numbers in the `This` column of the seed table), `status` ("keep"
        if the first word after `**Verdict:**` is "keep", else "drop") and `test` (the
        number after `**Test:**`, or None).
    """
    seeds, col = [], None
    for line in text.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        lower = [c.lower() for c in cells]
        if "this" in lower:
            col = lower.index("this")
        elif col is not None and cells[0].isdigit() and col < len(cells):
            value = _number(cells[col])
            if value is not None:
                seeds.append(value)
    verdict = re.search(r"\*\*Verdict:?\*\*:?\s*([A-Za-z]+)", text)
    test = re.search(r"\*\*Test:?\*\*:?\s*(.*)", text)
    return {
        "seeds": seeds,
        "status": "keep" if verdict and verdict[1].lower() == "keep" else "drop",
        "test": _number(test[1].split("(")[0]) if test else None,
    }


def read_reports(folder: Path = EXP_DIR) -> list[dict]:
    """Read every `NNN-category-name.md` report that has seed scores.

    Args:
        folder: the experiments folder.

    Returns:
        One dict per report, sorted by number, with `id`, `name` ("data-randaugment"),
        `status`, `seeds`, `mean`, `lo`, `hi` and `test`.
    """
    rows = []
    for path in folder.glob("[0-9][0-9][0-9]*-*.md"):
        m = REPORT.fullmatch(path.name)
        if not m:
            continue
        r = parse_report(path.read_text())
        if not r["seeds"]:
            continue
        rows.append({**r, "id": int(m[1]), "name": m[2], "mean": statistics.fmean(r["seeds"]),
                     "lo": min(r["seeds"]), "hi": max(r["seeds"])})
    return sorted(rows, key=lambda r: r["id"])


def baseline_steps(rows: list[dict]) -> list[tuple[int, float]]:
    """The baseline over time: `(experiment number, score)` at each kept experiment.

    Args:
        rows: reports from `read_reports`.

    Returns:
        One point per kept experiment, in order. The first report counts as kept.
    """
    return [(r["id"], r["mean"]) for i, r in enumerate(rows) if i == 0 or r["status"] == "keep"]


def keep_bar(rows: list[dict]) -> float | None:
    """How much a new experiment must gain over the baseline to be kept.

    Args:
        rows: reports from `read_reports`; the first is the reference baseline.

    Returns:
        `KEEP_FACTOR * sqrt(2) * seed std` of the first report, or None with < 2 seeds.
    """
    if not rows or len(rows[0]["seeds"]) < 2:
        return None
    return KEEP_FACTOR * math.sqrt(2) * statistics.stdev(rows[0]["seeds"])


def place_labels(fig, ax, labels: list[tuple[dict, str]], t: dict,
                 avoid: list[tuple[float, float]] = ()) -> None:
    """Label the kept experiments without overlap.

    Each label starts just above and right of its dot. If its text box would overlap a
    label already placed, the legend, other text in the axes, or a marker in `avoid`, it
    moves up one line at a time, and a thin leader line joins it to its dot.

    Args:
        fig: the figure, drawn once here to measure text.
        ax: the axes.
        labels: `(report, text)` pairs, left to right.
        t: theme tokens.
        avoid: data points (test reads) that a label must not cover.
    """
    renderer = fig.canvas.get_renderer()
    px = fig.dpi / 72                      # pixels per point
    taken = [a.get_window_extent(renderer) for a in ax.texts]
    if ax.get_legend() is not None:
        taken.append(ax.get_legend().get_window_extent(renderer))
    for x, y in avoid:
        cx, cy = ax.transData.transform((x, y))
        taken.append(Bbox.from_bounds(cx - 9, cy - 9, 18, 18))
    dx, step, gap = 6, 11, 2               # points
    for r, text in labels:
        probe = ax.text(0, 0, text, fontsize=8.5)
        size = probe.get_window_extent(renderer)
        probe.remove()
        x0, y0 = ax.transData.transform((r["id"], r["mean"]))
        dy = 8
        for _ in range(15):
            box = Bbox.from_bounds(x0 + dx * px - gap, y0 + dy * px - gap,
                                   size.width + 2 * gap, size.height + 2 * gap)
            if not any(box.overlaps(b) for b in taken):
                break
            dy += step
        taken.append(box)
        moved = dy > 8
        ax.annotate(text, (r["id"], r["mean"]), xytext=(dx, dy), textcoords="offset points",
                    ha="left", va="bottom", fontsize=8.5, color=t["text2"], zorder=5,
                    path_effects=[patheffects.withStroke(linewidth=3, foreground=t["surface"])],
                    arrowprops={"arrowstyle": "-", "color": t["axis"], "lw": 0.8,
                                "shrinkA": 0, "shrinkB": 4} if moved else None)


def draw(rows: list[dict], out: Path, theme: str = "light", title: str | None = None) -> Path:
    """Draw the chart and write it to `out`.

    Args:
        rows: reports from `read_reports`, at least one.
        out: PNG path to write.
        theme: "light" or "dark".
        title: chart title. Default: "Progress: N experiments, K kept".

    Returns:
        `out`.
    """
    t = THEMES[theme]
    plt.rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 10})
    fig, ax = plt.subplots(figsize=(13, 6.2), dpi=150)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])

    steps = baseline_steps(rows)
    last_id = rows[-1]["id"]
    xs = [s[0] for s in steps] + [last_id + 0.5]
    ys = [s[1] for s in steps] + [steps[-1][1]]

    bar = keep_bar(rows)
    if bar:
        ax.fill_between(xs, ys, [y + bar for y in ys], step="post", color=t["band"],
                        linewidth=0, zorder=1)
    ax.step(xs, ys, where="post", color=t["kept"], linewidth=2, zorder=3,
            solid_joinstyle="round", solid_capstyle="round")

    labels = []
    for i, r in enumerate(rows):
        kept = i == 0 or r["status"] == "keep"
        colour = t["kept"] if kept else t["dropped"]
        ax.plot([r["id"], r["id"]], [r["lo"], r["hi"]], color=colour, linewidth=1,
                alpha=0.55 if kept else 0.9, zorder=2, solid_capstyle="round")
        ax.scatter(r["id"], r["mean"], s=46 if kept else 26, color=colour,
                   edgecolors=t["surface"], linewidths=1.5, zorder=4)
        if kept:
            labels.append((r, "baseline" if i == 0 else f"{r['id']:03d} {r['name']}"))
        if r["test"] is not None:
            ax.scatter(r["id"], r["test"], s=52, marker="^", facecolors=t["surface"],
                       edgecolors=t["test"], linewidths=1.6, zorder=4)
            ax.annotate(f"test {r['test']:.3f}", (r["id"], r["test"]), xytext=(0, -14),
                        textcoords="offset points", ha="center", fontsize=8, color=t["muted"])

    n_kept = sum(1 for i, r in enumerate(rows) if i > 0 and r["status"] == "keep")
    ax.set_title(title or f"Progress: {len(rows) - 1} experiments, {n_kept} kept",
                 loc="left", fontsize=14, color=t["text"], pad=26, fontweight="bold")
    ax.text(0, 1.015, "Validation macro F1 (present species), mean of seeds; whisker = seed "
            "range. Higher is better.", transform=ax.transAxes, fontsize=9.5, color=t["text2"])
    ax.set_xlabel("experiment (PR number)", color=t["muted"])
    ax.set_ylabel("val macro F1 (present)", color=t["muted"])
    ax.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], labelsize=9)
    ax.set_xlim(rows[0]["id"] - 1, last_id + 1.5)
    lows = [r["lo"] for r in rows] + [r["test"] for r in rows if r["test"] is not None]
    highs = [r["hi"] for r in rows] + [y + (bar or 0) for y in ys]
    pad = 0.08 * (max(highs) - min(lows) or 0.01)
    ax.set_ylim(min(lows) - pad, max(highs) + 3 * pad)

    handles = [
        Line2D([], [], marker="o", linestyle="", color=t["kept"], markersize=7, label="kept"),
        Line2D([], [], marker="o", linestyle="", color=t["dropped"], markersize=6, label="dropped"),
        Line2D([], [], color=t["kept"], linewidth=2, label="current baseline"),
        Line2D([], [], marker="^", linestyle="", markerfacecolor=t["surface"],
               markeredgecolor=t["test"], markersize=7, label="test read (milestone)"),
    ]
    if bar:
        handles.insert(3, Patch(facecolor=t["band"], label=f"keep bar (+{bar:.3f})"))
    leg = ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(t["text2"])

    fig.tight_layout()
    place_labels(fig, ax, labels, t, [(r["id"], r["test"]) for r in rows if r["test"] is not None])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=t["surface"], bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return out


def main() -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=EXP_DIR, help="folder of NNN-category-name.md")
    ap.add_argument("--out", type=Path, help="PNG to write (default: <dir>/progress.png)")
    ap.add_argument("--dark", action="store_true", help="dark background, for dark slides")
    ap.add_argument("--title", help="override the chart title")
    a = ap.parse_args()

    rows = read_reports(a.dir)
    if not rows:
        print(f"no report with seed scores in {a.dir}; nothing to draw")
        return
    out = draw(rows, a.out or a.dir / "progress.png", "dark" if a.dark else "light", a.title)
    best = baseline_steps(rows)[-1]
    print(f"wrote {out}\ncurrent baseline: {best[0]:03d}, val macro F1 {best[1]:.4f}")


if __name__ == "__main__":
    main()
