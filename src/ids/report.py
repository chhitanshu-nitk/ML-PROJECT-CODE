from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = [
    "accuracy", "balanced_accuracy", "precision", "recall", "f1",
    "f1_weighted", "auc", "mcc",
]


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(lines)


def write_reports(df: pd.DataFrame, out_dir: Path, protocol: str) -> pd.DataFrame:
    out_dir = Path(out_dir)
    g = df.groupby(["task", "mode", "model"])
    mean = g[METRICS + ["fit_s", "pred_s"]].mean()
    std = g[METRICS].std().fillna(0.0).add_suffix("_std")
    summary = pd.concat([mean, std], axis=1).reset_index()
    summary.to_csv(out_dir / "summary.csv", index=False)

    comp = summary[["task", "mode", "model"]].copy()
    comp["acc_%"] = (summary["accuracy"] * 100).round(2)
    comp["balanced_acc_%"] = (summary["balanced_accuracy"] * 100).round(2)
    comp["macro_precision_%"] = (summary["precision"] * 100).round(2)
    comp["macro_recall_%"] = (summary["recall"] * 100).round(2)
    comp["macro_f1_%"] = (summary["f1"] * 100).round(2)
    comp["weighted_f1_%"] = (summary["f1_weighted"] * 100).round(2)
    comp["auc_%"] = (summary["auc"] * 100).round(2)
    comp["mcc"] = summary["mcc"].round(4)
    comp.to_csv(out_dir / "summary_pct.csv", index=False)

    md = [f"# Results (protocol = `{protocol}`)\n",
          "Metrics are means over folds (macro-averaged precision / recall / F1). "
          "`mode=baseline` is the paper's 'All Features', `mode=proposal` is RO + SFE + PCA.\n"]
    for task in summary["task"].unique():
        t = summary[summary["task"] == task]
        tbl = t[["mode", "model"] + METRICS].copy()
        for c in METRICS:
            if c == "mcc":
                tbl[c] = tbl[c].round(4)
                continue
            tbl[c] = (tbl[c] * 100).round(2)
        md += [f"## {task}\n", _md(tbl), ""]
    (out_dir / "results.md").write_text("\n".join(md), encoding="utf-8")

    # bar chart like Fig. 15
    fig_dir = out_dir / "figures"; fig_dir.mkdir(exist_ok=True)
    for task in summary["task"].unique():
        t = summary[summary["task"] == task]
        models = list(dict.fromkeys(t["model"]))
        modes = list(dict.fromkeys(t["mode"]))
        fig, ax = plt.subplots(figsize=(6, 4))
        w = 0.8 / max(len(modes), 1)
        for k, m in enumerate(modes):
            vals = [t[(t["model"] == mod) & (t["mode"] == m)]["accuracy"].mean() * 100 for mod in models]
            ax.bar(np.arange(len(models)) + k * w, vals, w, label=m)
        ax.set_xticks(np.arange(len(models)) + w * (len(modes) - 1) / 2); ax.set_xticklabels(models)
        lo = max(0, np.nanmin(t["accuracy"]) * 100 - 2)
        ax.set_ylim(lo, 100.2); ax.set_ylabel("Accuracy (%)"); ax.set_title(f"{task} accuracy ({protocol})")
        ax.legend(); fig.tight_layout(); fig.savefig(fig_dir / f"accuracy_{task}.png", dpi=130); plt.close(fig)
    return comp
