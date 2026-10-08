"""The figures in ``results/``, drawn from the stage reports.

``hawk-research plots`` regenerates them. Each figure answers one question from the
findings and reads only report JSON, so it can be redrawn without rerunning anything.
Requires matplotlib (``pip install -e ".[plots]"``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from hawk_research.data import paths

log = logging.getLogger(__name__)

BPS = 1e4
ROUND1 = "#4C72B0"
ROUND2 = "#DD8452"


def _load(rel: str) -> Optional[Dict[str, Any]]:
    f = paths.RESEARCH_DIR / rel
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"figure.dpi": 110, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
    )
    return plt


def zone_vs_control(out: Path) -> Optional[Path]:
    """Zone minus matched control: direction carries nothing, volatility a little."""
    r1, r2 = _load("reports/null_baselines.json"), _load("reports/null_baselines_smc.json")
    if r1 is None:
        return None
    plt = _plt()
    rows = [
        ("directional_signal_4h", "signed return, 4h"),
        ("directional_signal_24h", "signed return, 24h"),
        ("volatility_filter_4h", "forward realised vol, 4h"),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for i, (key, _label) in enumerate(rows):
        for j, (rep, colour, name) in enumerate(
            ((r1, ROUND1, "round 1 (wick candles)"), (r2, ROUND2, "round 2 (SMC order blocks)"))
        ):
            if rep is None or key not in rep.get("roles", {}):
                continue
            s = rep["roles"][key]
            y = i + (0.15 if j == 0 else -0.15)
            ax.errorbar(
                s["diff"] * BPS,
                y,
                xerr=[[(s["diff"] - s["ci_low"]) * BPS], [(s["ci_high"] - s["diff"]) * BPS]],
                fmt="o",
                color=colour,
                capsize=3,
                label=name if i == 0 else None,
            )
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(len(rows)), [r[1] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("zone minus matched control (bps), 95% block-bootstrap interval")
    ax.set_title("Zones vs volume- and volatility-matched non-zone bars")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    return _save(fig, out / "zone_vs_matched_control.png")


def walk_forward(out: Path) -> Optional[Path]:
    """Per-fold out-of-sample mean of the direction rule, gross and net."""
    reps = [
        (r, n, c)
        for r, n, c in (
            (_load("reports/null_baselines.json"), "round 1", ROUND1),
            (_load("reports/null_baselines_smc.json"), "round 2", ROUND2),
        )
        if r
    ]
    if not reps:
        return None
    plt = _plt()
    fig, axes = plt.subplots(len(reps), 1, figsize=(7.4, 2.2 * len(reps)), sharex=False, squeeze=False)
    for ax, (rep, name, colour) in zip(axes[:, 0], reps):
        folds = rep["walk_forward"]["folds"]
        x = list(range(1, len(folds) + 1))
        gross = [f["test_mean_gross"] * BPS for f in folds]
        net = [f["test_mean_net"] * BPS for f in folds]
        ax.bar([v - 0.2 for v in x], gross, width=0.4, color=colour, alpha=0.45, label="gross")
        ax.bar([v + 0.2 for v in x], net, width=0.4, color=colour, label="net (MEXC futures)")
        ax.axhline(0, color="black", lw=0.8)
        s = rep["walk_forward"]["summary"]
        ax.set_title(f"{name}: {s['folds_net_positive']} of {s['n_folds']} test months net-positive")
        ax.set_ylabel("bps / trade")
        ax.set_xticks(x[::2], [folds[i - 1]["test_start"][:7] for i in x[::2]], rotation=45, ha="right")
        ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    return _save(fig, out / "walk_forward_folds.png")


def net_by_configuration(out: Path) -> Optional[Path]:
    """Every logged configuration, net of MEXC futures costs, with its interval."""
    r1, r2 = _load("reports/null_baselines.json"), _load("reports/smc_eval.json")
    items: List[tuple] = []
    for rep, key, colour in ((r1, "strategy_evaluations", ROUND1), (r2, "strategies", ROUND2)):
        if not rep:
            continue
        for label, s in rep[key].items():
            if not label.endswith("mexc_futures_taker") or s.get("mean_ret_net") is None:
                continue
            lo, hi = s.get("ci_low_cross"), s.get("ci_high_cross")
            items.append(
                (
                    label.replace("|mexc_futures_taker", ""),
                    s["mean_ret_net"] * BPS,
                    None if lo is None else lo * BPS,
                    None if hi is None else hi * BPS,
                    colour,
                )
            )
    if not items:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 0.22 * len(items) + 1.2))
    for i, (_label, m, lo, hi, colour) in enumerate(items):
        err = None if lo is None else [[m - lo], [hi - m]]
        ax.errorbar(m, i, xerr=err, fmt="o", color=colour, ms=3, capsize=2)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(len(items)), [it[0] for it in items], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("net mean per trade (bps), MEXC futures costs, 95% interval")
    fig.suptitle("No configuration makes money after costs\n(blue: round 1, orange: round 2)", fontsize=9)
    fig.tight_layout()
    return _save(fig, out / "net_by_configuration.png")


def harness_calibration(out: Path) -> Optional[Path]:
    """The harness finds a planted edge, and its zero-null intervals are calibrated."""
    rep = _load("reports/harness_controls.json")
    if rep is None:
        return None
    plt = _plt()
    perm = rep["negative_controls"]["shuffled_labels_permutation"]
    pos = rep["positive_control"]
    obs = rep["observed_reference"]
    rows = [
        ("injected +25 bps edge, measured", pos["measured_mean"] * BPS, [c * BPS for c in pos["ci"]]),
        (
            "permutation null (mean ± 2 sd)",
            perm["null_mean"] * BPS,
            [
                (perm["null_mean"] - 2 * perm["null_std"]) * BPS,
                (perm["null_mean"] + 2 * perm["null_std"]) * BPS,
            ],
        ),
        ("observed: zone direction, 4h gross", obs["mean_return_gross"] * BPS, [c * BPS for c in obs["ci"]]),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    for i, (_label, m, (lo, hi)) in enumerate(rows):
        ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="o", color=ROUND1, capsize=3)
    ax.axvline(0, color="black", lw=0.8)
    ax.axvline(25, color="grey", lw=0.8, ls="--")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    rs = rep["negative_controls"]["randomized_sides"]
    fig.suptitle(
        f"Harness controls: random sides flagged {rs['significant_at_95pct']} of {rs['repeats']} times\n"
        f"({rs['false_positive_rate']:.1%} at a nominal 5%)",
        fontsize=9,
    )
    ax.set_xlabel("mean per trade (bps)")
    return _save(fig, out / "harness_controls.png")


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, metadata={"Software": None})  # no version string: identical bytes across runs
    import matplotlib.pyplot as plt

    plt.close(fig)
    log.info("wrote %s", path)
    return path


FIGURES = (zone_vs_control, walk_forward, net_by_configuration, harness_calibration)


def main(results_dir: Path) -> List[Path]:
    written = []
    for fn in FIGURES:
        p = fn(Path(results_dir))
        if p is None:
            log.warning("skip %s: its report has not been generated", fn.__name__)
        else:
            written.append(p)
    return written
