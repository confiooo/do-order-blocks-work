"""The figures in ``results/``, drawn from the stage reports.

``hawk-research plots`` regenerates them. Each figure answers one question from the
findings and reads only report JSON, so it can be redrawn without rerunning anything.
Every figure is written twice, ``<name>.png`` for light backgrounds and
``<name>-dark.png`` for dark ones. Requires matplotlib (``pip install -e ".[plots]"``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from hawk_research.data import paths

log = logging.getLogger(__name__)

BPS = 1e4

# Colours: two categorical slots (round 1, round 2) and the chart chrome, per theme.
# The two series colours were checked for colour-blind separation and contrast
# against each surface.
THEMES: Dict[str, Dict[str, str]] = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "band": "#f0efec",
        "round1": "#2a78d6",
        "round2": "#eb6834",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "band": "#242423",
        "round1": "#3987e5",
        "round2": "#d95926",
    },
}
ROUND_LABEL = {"round1": "Round 1: original detector", "round2": "Round 2: textbook order blocks"}
MARKER = {"round1": "o", "round2": "s"}


def _load(rel: str) -> Optional[Dict[str, Any]]:
    f = paths.RESEARCH_DIR / rel
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def _plt(t: Dict[str, str]) -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 200,
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "figure.facecolor": t["surface"],
            "axes.facecolor": t["surface"],
            "savefig.facecolor": t["surface"],
            "axes.edgecolor": t["axis"],
            "axes.labelcolor": t["ink2"],
            "axes.titlecolor": t["ink"],
            "text.color": t["ink"],
            "xtick.color": t["muted"],
            "ytick.color": t["muted"],
            "xtick.labelcolor": t["ink2"],
            "ytick.labelcolor": t["ink2"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.grid": False,
            "legend.frameon": False,
        }
    )
    return plt


def _heading(fig: Any, t: Dict[str, str], title: str, subtitle: str) -> None:
    """Left-aligned title (the finding) and a subtitle (what is plotted), placed in inches from the top."""
    h = fig.get_figheight()
    fig.text(0.02, 1 - 0.2 / h, title, ha="left", va="top", fontsize=12.5, fontweight="bold", color=t["ink"])
    fig.text(0.02, 1 - 0.5 / h, subtitle, ha="left", va="top", fontsize=9, color=t["ink2"])


def _num(v: float) -> str:
    return f"{v:+.1f}".replace("-", "\u2212")


def _xgrid(ax: Any, t: Dict[str, str]) -> None:
    ax.grid(axis="x", color=t["grid"], lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.spines["bottom"].set_visible(False)


def _zero(ax: Any, t: Dict[str, str], label: Optional[str] = None) -> None:
    ax.axvline(0, color=t["ink2"], lw=1.0, zorder=1)
    if label:
        ax.text(
            0,
            1.0,
            f" {label}",
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=7.5,
            color=t["ink2"],
        )


def _interval(
    ax: Any, t: Dict[str, str], x: float, lo: float, hi: float, y: float, key: str, size: float = 7.5
) -> None:
    """A point estimate with its interval: a 2px line and a ringed marker."""
    c = t[key]
    ax.plot([lo, hi], [y, y], color=c, lw=2, solid_capstyle="round", zorder=2)
    ax.plot([x], [y], MARKER[key], color=c, ms=size, mec=t["surface"], mew=1.6, zorder=3)


def zone_vs_control(out: Path, t: Dict[str, str]) -> Optional[Tuple[Any, str]]:
    """Zone minus matched control: direction carries nothing, volatility a little."""
    r1, r2 = _load("reports/null_baselines.json"), _load("reports/null_baselines_smc.json")
    if r1 is None:
        return None
    plt = _plt(t)
    groups = [
        (
            "Direction",
            [
                ("directional_signal_4h", "Signed return, 4h"),
                ("directional_signal_24h", "Signed return, 24h"),
            ],
        ),
        ("Volatility", [("volatility_filter_4h", "Realised volatility, 4h")]),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 4.1))
    fig.subplots_adjust(left=0.27, right=0.97, top=1 - 1.2 / 4.1, bottom=0.14)
    yticks, ylabels, y = [], [], 0.0
    for gi, (group, rows) in enumerate(groups):
        if gi:
            y += 0.6
        ax.text(
            -0.36,
            y - 0.55,
            group.upper(),
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=7.5,
            fontweight="bold",
            color=t["muted"],
        )
        for key, label in rows:
            yticks.append(y)
            ylabels.append(label)
            for off, rep, series in ((-0.17, r1, "round1"), (0.17, r2, "round2")):
                if rep is None or key not in rep.get("roles", {}):
                    continue
                s = rep["roles"][key]
                d, lo, hi = s["diff"] * BPS, s["ci_low"] * BPS, s["ci_high"] * BPS
                _interval(ax, t, d, lo, hi, y + off, series)
                excludes = lo > 0 or hi < 0
                ax.text(
                    hi,
                    y + off,
                    f"  {_num(d)}" + ("  excludes 0" if excludes else ""),
                    va="center",
                    ha="left",
                    fontsize=7.5,
                    color=t["ink"] if excludes else t["ink2"],
                    fontweight="bold" if excludes else "normal",
                )
            y += 1.0
    ax.set_yticks(yticks, ylabels)
    ax.set_ylim(y - 0.4, -0.75)
    lo_all, hi_all = ax.get_xlim()
    ax.set_xlim(lo_all, hi_all + 0.22 * (hi_all - lo_all))
    _xgrid(ax, t)
    _zero(ax, t)
    ax.set_xlabel("Zone minus matched control, bps (95% block-bootstrap interval)", fontsize=8.5)
    handles = [
        ax.plot([], [], MARKER[k], color=t[k], ms=7, mec=t["surface"], mew=1.4, label=ROUND_LABEL[k])[0]
        for k in ("round1", "round2")
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.012, 1 - 0.78 / 4.1),
        ncol=2,
        fontsize=8.5,
        handletextpad=0.3,
        columnspacing=1.6,
        labelcolor=t["ink2"],
    )
    _heading(
        fig,
        t,
        "Zones carry no direction, only a little extra volatility",
        "Each zone compared with volume- and volatility-matched ordinary bars on the same symbol, ±24h",
    )
    return fig, "zone_vs_matched_control"


def _month(iso: str) -> str:
    import datetime as dt

    return dt.date.fromisoformat(iso[:10]).strftime("%b %y")


def walk_forward(out: Path, t: Dict[str, str]) -> Optional[Tuple[Any, str]]:
    """Per-fold out-of-sample mean of the direction rule, gross and net."""
    reps = [
        (r, s)
        for r, s in (
            (_load("reports/null_baselines.json"), "round1"),
            (_load("reports/null_baselines_smc.json"), "round2"),
        )
        if r
    ]
    if not reps:
        return None
    plt = _plt(t)
    height = 2.3 * len(reps) + 1.5
    fig, axes = plt.subplots(len(reps), 1, figsize=(8.2, height), sharex=True, sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.08, right=0.98, top=1 - 1.45 / height, bottom=0.6 / height, hspace=0.32)
    for ax, (rep, series) in zip(axes[:, 0], reps):
        folds = rep["walk_forward"]["folds"]
        x = list(range(len(folds)))
        net = [f["test_mean_net"] * BPS for f in folds]
        gross = [f["test_mean_gross"] * BPS for f in folds]
        ax.bar(x, net, width=0.72, color=t[series], zorder=2, label="Net of MEXC futures costs")
        ax.plot(x, gross, "_", color=t["ink2"], ms=10, mew=1.6, zorder=3, label="Gross (before costs)")
        ax.axhline(0, color=t["ink2"], lw=1.0, zorder=1)
        ax.grid(axis="y", color=t["grid"], lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        ax.spines["bottom"].set_visible(False)
        s = rep["walk_forward"]["summary"]
        ax.set_title(
            f"{ROUND_LABEL[series]}: {s['folds_net_positive']} of {s['n_folds']} months net-positive",
            loc="left",
            fontsize=9.5,
            fontweight="bold",
            color=t["ink"],
            pad=6,
        )
        ax.set_ylabel("bps per trade", fontsize=8)
        ax.set_xticks(x[::3], [_month(folds[i]["test_start"]) for i in x[::3]])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.012, 1 - 0.78 / height),
        ncol=2,
        fontsize=8.5,
        labelcolor=t["ink2"],
        handlelength=1.4,
        columnspacing=1.6,
    )
    _heading(
        fig,
        t,
        "Trading the zone's direction lost money almost every month",
        "Walk-forward test months (90 days train, 30 days test); mean return per trade of the direction rule",
    )
    return fig, "walk_forward_folds"


ENTRY = {"close": "close entry", "retest": "retest entry"}


def _config_label(parts: List[str]) -> str:
    """``direction|retest|barrier|24h|0.02/0.01`` -> ``Direction · retest entry · TP 2% / SL 1%``."""
    rule, entry, exit_, horizon = parts[:4]
    head = {
        "direction": "Direction",
        "fade": "Fade",
        "all": "All order blocks",
        "fvg": "With fair value gap",
    }[rule]
    if exit_ == "barrier":
        tp, sl = parts[4].split("/")
        tail = f"TP {float(tp):.0%} / SL {float(sl):.0%}"
    else:
        tail = f"{horizon} hold"
    return f"{head} · {ENTRY[entry]} · {tail}"


def net_by_configuration(out: Path, t: Dict[str, str]) -> Optional[Tuple[Any, str]]:
    """Every logged configuration, net of MEXC futures costs, with its interval."""
    panels: List[Tuple[str, List[Tuple[str, float, float, float]]]] = []
    for rel, key, series in (
        ("reports/null_baselines.json", "strategy_evaluations", "round1"),
        ("reports/smc_eval.json", "strategies", "round2"),
    ):
        rep = _load(rel)
        if not rep:
            continue
        rows = []
        for label, s in rep[key].items():
            parts = label.split("|")
            if parts[-1] != "mexc_futures_taker" or s.get("mean_ret_net") is None:
                continue
            if series == "round2":
                if parts[1] != "full":  # development/holdout splits re-read the same trial
                    continue
                parts = [parts[0], parts[2], parts[3], "24h", parts[4]]
            lo, hi = s.get("ci_low_cross"), s.get("ci_high_cross")
            if lo is None or hi is None:
                continue
            rows.append(
                (
                    _config_label(parts[:-1] if series == "round1" else parts),
                    s["mean_ret_net"] * BPS,
                    lo * BPS,
                    hi * BPS,
                )
            )
        rows.sort(key=lambda r: -r[1])
        panels.append((series, rows))
    if not panels:
        return None
    plt = _plt(t)
    n = [len(r) for _, r in panels]
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(7.8, 0.27 * sum(n) + 2.2),
        sharex=True,
        squeeze=False,
        gridspec_kw={"height_ratios": n},
    )
    height = 0.27 * sum(n) + 2.2
    fig.subplots_adjust(left=0.40, right=0.94, top=1 - 1.15 / height, bottom=0.6 / height, hspace=0.32)
    for ax, (series, rows) in zip(axes[:, 0], panels):
        for i, (_label, m, lo, hi) in enumerate(rows):
            _interval(ax, t, m, lo, hi, i, series, size=6.5)
        ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8)
        ax.set_ylim(len(rows) - 0.5, -0.7)
        ax.set_title(
            ROUND_LABEL[series], loc="left", fontsize=9.5, fontweight="bold", color=t["ink"], pad=4, x=-0.68
        )
        _xgrid(ax, t)
        _zero(ax, t, "break-even" if ax is axes[0, 0] else None)
    axes[-1, 0].set_xlabel("Net mean per trade, bps (MEXC futures costs, 95% interval)", fontsize=8.5)
    lo_all, _ = axes[-1, 0].get_xlim()
    axes[-1, 0].set_xlim(lo_all, 2.5)
    _heading(
        fig,
        t,
        "No configuration makes money after costs",
        "Every tested configuration over the full sample, sorted from best to worst",
    )
    return fig, "net_by_configuration"


def harness_calibration(out: Path, t: Dict[str, str]) -> Optional[Tuple[Any, str]]:
    """The harness finds a planted edge, and its zero-null intervals are calibrated."""
    rep = _load("reports/harness_controls.json")
    if rep is None:
        return None
    plt = _plt(t)
    perm = rep["negative_controls"]["shuffled_labels_permutation"]
    pos = rep["positive_control"]
    obs = rep["observed_reference"]
    rs = rep["negative_controls"]["randomized_sides"]
    edge = pos["injected_edge_bps"]
    rows = [
        (
            f"Planted +{edge:.0f} bps edge, as measured",
            pos["measured_mean"] * BPS,
            [c * BPS for c in pos["ci"]],
            "round1",
        ),
        (
            "Observed zone direction, 4h, gross",
            obs["mean_return_gross"] * BPS,
            [c * BPS for c in obs["ci"]],
            "round1",
        ),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 3.3))
    fig.subplots_adjust(left=0.30, right=0.97, top=1 - 1.25 / 3.3, bottom=0.17)
    null_lo = (perm["null_mean"] - 2 * perm["null_std"]) * BPS
    null_hi = (perm["null_mean"] + 2 * perm["null_std"]) * BPS
    ax.axvspan(null_lo, null_hi, color=t["band"], zorder=0)
    ax.text(
        (null_lo + null_hi) / 2,
        1.0,
        "permutation null\n(mean ± 2 sd)",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=t["ink2"],
    )
    ax.axvline(edge, color=t["muted"], lw=1.0, ls=(0, (3, 3)), zorder=1)
    ax.text(
        edge,
        1.0,
        f"planted\n+{edge:.0f} bps",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=t["ink2"],
    )
    for i, (_label, m, (lo, hi), series) in enumerate(rows):
        _interval(ax, t, m, lo, hi, i, series)
        ax.text(m, i - 0.16, _num(m), va="bottom", ha="center", fontsize=7.5, color=t["ink"])
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_ylim(len(rows) - 0.4, -0.6)
    lo_all, hi_all = ax.get_xlim()
    ax.set_xlim(lo_all, hi_all + 0.08 * (hi_all - lo_all))
    _xgrid(ax, t)
    _zero(ax, t)
    ax.set_xlabel("Mean return per trade, bps (95% cross-sectional interval)", fontsize=8.5)
    _heading(
        fig,
        t,
        "The test harness finds a planted edge and is not fooled by noise",
        f"With random trade directions, {rs['significant_at_95pct']} of {rs['repeats']} runs looked significant "
        f"({rs['false_positive_rate']:.1%}, nominal 5%)",
    )
    return fig, "harness_controls"


def _save(fig: Any, path: Path) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={"Software": None})  # no version string: identical bytes across runs
    plt.close(fig)
    log.info("wrote %s", path)
    return path


FIGURES: Tuple[Callable[[Path, Dict[str, str]], Optional[Tuple[Any, str]]], ...] = (
    zone_vs_control,
    walk_forward,
    net_by_configuration,
    harness_calibration,
)


def main(results_dir: Path) -> List[Path]:
    written = []
    out = Path(results_dir)
    for fn in FIGURES:
        for theme, suffix in (("light", ""), ("dark", "-dark")):
            made = fn(out, THEMES[theme])
            if made is None:
                log.warning("skip %s: its report has not been generated", fn.__name__)
                break
            fig, name = made
            written.append(_save(fig, out / f"{name}{suffix}.png"))
    return written
