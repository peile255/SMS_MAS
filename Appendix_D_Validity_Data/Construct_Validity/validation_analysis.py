# -*- coding: utf-8 -*-
"""
validation_analysis.py
对 classification_labels.csv 做一致性分析：
- 逐维度计算 Cohen's Kappa
- 绘制混淆矩阵
- 导出 metrics_per_dimension.csv
- 绘制 Kappa 柱状图
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, confusion_matrix


DIMENSIONS = [
    "PlatformType",
    "ApplicationDomain",
    "AgentRole",
    "EvaluationMethod",
    "Challenges",
]


def parse_args():
    p = argparse.ArgumentParser(description="Validation analysis for classification labels (R1 vs R2)")
    p.add_argument("-i", "--input", default="classification_labels.csv",
                   help="Input CSV (default: classification_labels.csv)")
    p.add_argument("--outdir", default="results", help="Output directory (default: results)")
    return p.parse_args()


def load_df(path):
    df = pd.read_csv(path)
    # 基本检查
    for dim in DIMENSIONS:
        for suffix in ["R1", "R2"]:
            col = f"{dim}_{suffix}"
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")
    return df


def compute_kappa(r1, r2):
    return cohen_kappa_score(r1, r2)


def save_confmat(cm, labels, title, outpath):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("R2")
    plt.ylabel("R1")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = load_df(args.input)

    rows = []
    kappas = []
    for dim in DIMENSIONS:
        r1 = df[f"{dim}_R1"].astype(str).str.strip()
        r2 = df[f"{dim}_R2"].astype(str).str.strip()

        kappa = compute_kappa(r1, r2)
        kappas.append(kappa)

        # 保证标签顺序固定
        labels = sorted(set(r1) | set(r2))
        cm = confusion_matrix(r1, r2, labels=labels)

        outpath = os.path.join(args.outdir, f"confmat_{dim}.png")
        save_confmat(cm, labels, f"Confusion Matrix - {dim}", outpath)

        rows.append({
            "Dimension": dim,
            "NumClasses": len(labels),
            "Kappa": kappa,
            "N": len(r1)
        })

    # 保存结果表
    res = pd.DataFrame(rows).sort_values("Dimension")
    res.to_csv(os.path.join(args.outdir, "metrics_per_dimension.csv"), index=False, encoding="utf-8")

    # 画 κ 柱状图
    plt.figure(figsize=(8, 4.8))
    sns.barplot(x="Dimension", y="Kappa", data=res)
    plt.axhline(0.78, linestyle="--", color="blue")
    plt.axhline(0.86, linestyle="--", color="blue")
    plt.ylim(0.5, 1.0)
    plt.title("Cohen's Kappa per Dimension (R1 vs R2)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "kappa_bar.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print("\n=========== Classification Agreement Results ===========")
    print(res.to_string(index=False, formatters={"Kappa": "{:.3f}".format}))
    print("--------------------------------------------------------")
    print(f"Macro-average κ: {res['Kappa'].mean():.3f}")
    print(f"Saved results in: {args.outdir}")
    print("========================================================\n")


if __name__ == "__main__":
    main()
