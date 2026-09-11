#!/usr/bin/env python3
"""
Regenerate Fig. log-arrival-rate for PLAP §3.2.2:
  (a) HDFS: log counts per minute
  (b) BlueGene/L: log counts per hour

Lives next to the time-window prediction experiment:
  plap-main/log_anomaly_prediction/logprompt/log3p_prediction/experiment_window_size.py

Supports:
  1) Raw Loghub logs: HDFS.log / BGL.log
  2) Drain structured CSVs: HDFS.log_structured.csv / BGL.log_structured.csv

Example (server; no args needed if default paths exist):
    cd /tmp/pycharm_project_7c6491fa/plap-main
    /home/zhouzt/ccccccconda_exp/bin/python \\
        log_anomaly_prediction/logprompt/log3p_prediction/plot_log_arrival_rate.py
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HDFS_RE = re.compile(r"^(\d{6})\s+(\d{6})\b")
BGL_RE = re.compile(r"(\d{4}-\d{2}-\d{2})-(\d{2})\.(\d{2})\.(\d{2})(?:\.\d+)?")

DEFAULT_HDFS = Path("/home/zhouzt/experiment/log_parsing/logs/HDFS/HDFS_v1/HDFS.log")
DEFAULT_BGL = Path("/home/zhouzt/experiment/log_parsing/logs/BGL/BGL/BGL.log")
SERVER_FIGS = Path("/home/zhouzt/experiment/figs")

SCRIPT_DIR = Path(__file__).resolve().parent
PLAP_ROOT = SCRIPT_DIR.parents[3]  # .../plap-main
WORKSPACE_ROOT = PLAP_ROOT.parent


def default_out_path() -> Path:
    if Path("/home/zhouzt/experiment").is_dir():
        return SERVER_FIGS / "fig-log-arrival-rate.png"
    return WORKSPACE_ROOT / "figs" / "fig-log-arrival-rate.png"


DEFAULT_OUT = default_out_path()
TICK_FONTSIZE = 20
FIGSIZE = (12, 5.2)


def _is_csv(path: Path) -> bool:
    return path.suffix.lower() == ".csv"


def count_hdfs_minutes(path: Path, max_lines: int | None = None) -> Counter:
    counts: Counter = Counter()
    if _is_csv(path):
        usecols = ["Date", "Time"]
        reader = pd.read_csv(path, usecols=usecols, chunksize=500_000)
        n = 0
        for chunk in reader:
            date = chunk["Date"].map(lambda x: f"{int(x):06d}")
            time = chunk["Time"].astype(str).str.zfill(6)
            dt = pd.to_datetime(date + time, format="%y%m%d%H%M%S", errors="coerce")
            dt = dt.dropna()
            for t in dt:
                counts[t.to_pydatetime().replace(second=0, microsecond=0)] += 1
            n += len(chunk)
            if max_lines is not None and n >= max_lines:
                break
        return counts

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            m = HDFS_RE.match(line)
            if not m:
                continue
            try:
                dt = datetime.strptime(m.group(1) + m.group(2), "%y%m%d%H%M%S")
            except ValueError:
                continue
            counts[dt.replace(second=0, microsecond=0)] += 1
    return counts


def count_bgl_hours(path: Path, max_lines: int | None = None) -> Counter:
    counts: Counter = Counter()
    if _is_csv(path):
        peek = pd.read_csv(path, nrows=2)
        candidates = [
            c
            for c in ["Time", "DateTime", "Timestamp", "datetime", "timestamp"]
            if c in peek.columns
        ]
        if not candidates:
            raise ValueError(f"BGL CSV has no time column. Columns={list(peek.columns)}")
        col = candidates[0]
        reader = pd.read_csv(path, usecols=[col], chunksize=500_000)
        n = 0
        for chunk in reader:
            raw = chunk[col].astype(str)
            normalized = raw.str.replace(
                r"^(\d{4}-\d{2}-\d{2})-(\d{2})\.(\d{2})\.(\d{2})",
                r"\1 \2:\3:\4",
                regex=True,
            )
            dt = pd.to_datetime(normalized, errors="coerce").dropna()
            for t in dt:
                counts[t.to_pydatetime().replace(minute=0, second=0, microsecond=0)] += 1
            n += len(chunk)
            if max_lines is not None and n >= max_lines:
                break
        return counts

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            m = BGL_RE.search(line)
            if not m:
                continue
            try:
                dt = datetime.strptime(
                    f"{m.group(1)} {m.group(2)}:{m.group(3)}:{m.group(4)}",
                    "%Y-%m-%d %H:%M:%S",
                )
            except ValueError:
                continue
            counts[dt.replace(minute=0, second=0, microsecond=0)] += 1
    return counts


def densify(counts: Counter, step: timedelta) -> tuple[np.ndarray, np.ndarray]:
    if not counts:
        raise ValueError("No timestamps parsed; check path / format.")
    keys = sorted(counts)
    start, end = keys[0], keys[-1]
    xs, ys = [], []
    t = start
    i = 0
    while t <= end:
        xs.append(i)
        ys.append(counts.get(t, 0) / 1000.0)  # thousands
        t += step
        i += 1
    return np.asarray(xs), np.asarray(ys)


def _save_panel(x, y, xlabel: str, ylabel: str, caption: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(x, y, color="#1f77b4", linewidth=0.8)
    ax.set_xlabel(xlabel, fontsize=TICK_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=TICK_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.text(
        0.5,
        0.01,
        caption,
        ha="center",
        va="bottom",
        fontsize=TICK_FONTSIZE,
        fontweight="bold",
    )
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    print(f"Saved: {out.with_suffix('.pdf')}")


def plot(hdfs_c: Counter, bgl_c: Counter, out: Path) -> None:
    hx, hy = densify(hdfs_c, timedelta(minutes=1))
    bx, by = densify(bgl_c, timedelta(hours=1))

    out = Path(out)
    hdfs_out = out.with_name(f"{out.stem}-hdfs{out.suffix}")
    bgl_out = out.with_name(f"{out.stem}-bgl{out.suffix}")

    _save_panel(
        hx,
        hy,
        "Time (minute)",
        "Number of Logs (thousands)",
        "(a) HDFS",
        hdfs_out,
    )
    _save_panel(
        bx,
        by,
        "Time (hour)",
        "Number of Logs (thousands)",
        "(b) BlueGene/L",
        bgl_out,
    )
    print(f"HDFS minutes={len(hx)}, peak={hy.max() * 1000:.0f} logs/min")
    print(f"BGL  hours={len(bx)}, peak={by.max() * 1000:.0f} logs/hour")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot HDFS/BGL log arrival rates.")
    parser.add_argument("--hdfs", type=Path, default=DEFAULT_HDFS, help="HDFS.log or structured CSV")
    parser.add_argument("--bgl", type=Path, default=DEFAULT_BGL, help="BGL.log or structured CSV")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="输出路径前缀。实际写成 fig-log-arrival-rate-hdfs.png 和 -bgl.png",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="Optional cap for a quick smoke test",
    )
    args = parser.parse_args()

    print(f"[1/3] Counting HDFS per minute from {args.hdfs} ...")
    hdfs_c = count_hdfs_minutes(args.hdfs, args.max_lines)
    print(f"      non-empty minute bins: {len(hdfs_c)}")

    print(f"[2/3] Counting BGL per hour from {args.bgl} ...")
    bgl_c = count_bgl_hours(args.bgl, args.max_lines)
    print(f"      non-empty hour bins: {len(bgl_c)}")

    print("[3/3] Plotting ...")
    plot(hdfs_c, bgl_c, args.out)


if __name__ == "__main__":
    main()
