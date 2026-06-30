#!/usr/bin/env python3
"""Plot rescored metrics vs experiment number (chronological attempt order).

Joins rescore_dev500.csv with the run's attempt timestamps so the x-axis is the
real progression of the run (1..N in time order), and draws AUROC (primary), AP,
old_mSR and new_mSR. Best-so-far AUROC and the random baselines are overlaid.

    uv run --with matplotlib python auto_rl_selection/plot_rescore.py \
        [--run results/rl-environment-selection/2026-06-26_033844] \
        [--csv auto_rl_selection/rescore_dev500.csv] \
        [--out auto_rl_selection/rescore_plot.png]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/rl-environment-selection/2026-06-26_033844")
    ap.add_argument("--csv", default="auto_rl_selection/rescore_dev500.csv")
    ap.add_argument("--out", default="auto_rl_selection/rescore_plot.png")
    args = ap.parse_args()

    attempts_dir = (REPO_ROOT / args.run / ".coral/public/attempts").resolve()
    # commit[:12] -> timestamp
    ts: dict[str, str] = {}
    for jf in attempts_dir.glob("*.json"):
        a = json.loads(jf.read_text())
        h = (a.get("commit_hash") or "")[:12]
        if h and a.get("timestamp"):
            ts[h] = a["timestamp"]

    rows = []
    for r in csv.DictReader(open(REPO_ROOT / args.csv)):
        if r["error"]:
            continue
        h = r["commit"]
        rows.append(
            {
                "commit": h,
                "t": ts.get(h, ""),
                "AUROC": float(r["AUROC"]),
                "AP": float(r["AP"]),
                "old_mSR": float(r["old_mSR"]),
                "new_mSR": float(r["new_mSR"]),
            }
        )
    # chronological order (fall back to CSV order if a timestamp is missing)
    rows.sort(key=lambda r: (r["t"] == "", r["t"]))
    x = list(range(1, len(rows) + 1))

    # best-so-far AUROC
    best = []
    cur = -1.0
    for r in rows:
        cur = max(cur, r["AUROC"])
        best.append(cur)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, [r["AUROC"] for r in rows], lw=1.4, color="#1f77b4", label="AUROC (primary)")
    ax.plot(x, best, lw=2.0, color="#1f77b4", ls="--", alpha=0.7, label="AUROC best-so-far")
    ax.plot(x, [r["AP"] for r in rows], lw=1.0, color="#2ca02c", alpha=0.8, label="AP")
    ax.plot(x, [r["new_mSR"] for r in rows], lw=1.0, color="#ff7f0e", alpha=0.8, label="mSR  [v=p(1-p)]")
    ax.plot(x, [r["old_mSR"] for r in rows], lw=1.0, color="#9467bd", alpha=0.6, label="old mSR  [v=1-p^8-(1-p)^8]")

    ax.axhline(0.50, color="#1f77b4", ls=":", lw=1, alpha=0.5)
    ax.text(len(rows), 0.502, " AUROC random=0.50", color="#1f77b4", fontsize=8, va="bottom", ha="right")
    ax.axhline(0.562, color="#2ca02c", ls=":", lw=1, alpha=0.5)
    ax.text(len(rows), 0.564, " AP random≈0.562", color="#2ca02c", fontsize=8, va="bottom", ha="right")

    # mark best AUROC attempt
    bi = max(range(len(rows)), key=lambda i: rows[i]["AUROC"])
    ax.scatter([bi + 1], [rows[bi]["AUROC"]], color="#d62728", zorder=5, s=40)
    ax.annotate(
        f"best AUROC={rows[bi]['AUROC']:.4f}\n{rows[bi]['commit']}",
        (bi + 1, rows[bi]["AUROC"]),
        textcoords="offset points", xytext=(6, 10), fontsize=8, color="#d62728",
    )

    ax.set_xlabel("experiment number (chronological attempt order)")
    ax.set_ylabel("score")
    ax.set_title("rl-env-selection run 2026-06-26 — rescored on dev500 (163 attempts)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out = REPO_ROOT / args.out
    fig.savefig(out, dpi=130)
    print(f"wrote {out}  ({len(rows)} attempts, {sum(1 for r in rows if r['t'])} with timestamps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
