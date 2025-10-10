# -*- coding: utf-8 -*-
"""
validation_analysis.py  (Micro F1 主结果版 · 水平渐变条形图样式)

- 自动检测当前目录的 classification_labels.csv（稳健编码与分隔符尝试）
- 按多标签任务计算：Micro F1（主结果）、Macro F1、Subset accuracy、Avg Jaccard、Hamming score
- 导出 metrics_per_dimension.csv（Micro F1 为首列）
- 绘制 micro_f1_bar.png（水平条形图，渐变配色，右侧标注）

Usage:
    python validation_analysis.py
"""

import os
import sys
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Set, Tuple
from sklearn.metrics import f1_score, hamming_loss

# -----------------------
# 全局画图风格
# -----------------------
plt.rcParams['font.family'] = 'Times New Roman'

# -----------------------
# 参数设置
# -----------------------
CSV_NAME = "classification_labels.csv"
OUTPUT_DIR = "results_validation"

# 注意：你的列名第一维是 Platform_Typ（不是 Platform_Type）
DIMENSIONS = [
    "Platform_Typ",
    "Agent_Role",
    "Application_Domain",
    "Evaluation_Method",
    "Metrics",
    "Challenges",
]

# -----------------------
# 工具函数
# -----------------------
def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def is_list_like_string(s) -> bool:
    return isinstance(s, str) and s.strip().startswith("[") and s.strip().endswith("]")

def tokens_from_cell(v) -> Set[str]:
    """将单元格解析为标签集合（用于多标签度量）"""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return set()
    s = str(v).strip()
    if s == "":
        return set()
    try:
        if is_list_like_string(s):
            parsed = ast.literal_eval(s)
            if not isinstance(parsed, (list, tuple)):
                parsed = [str(parsed)]
            return set(str(x).strip() for x in parsed if str(x).strip())
        else:
            return {s}
    except Exception:
        return {s}

def read_csv_safely(path: str) -> Tuple[pd.DataFrame, str]:
    """尝试多种编码与分隔符读 CSV，最大化兼容性。"""
    encodings = ["utf-8", "utf-8-sig", "gb18030", "cp936", "latin1"]
    seps = [",", "\t"]
    last_err = None
    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep, engine="python")
                if df.shape[1] == 1 and sep == ",":
                    df = pd.read_csv(path, encoding=enc, sep="\t", engine="python")
                return df, enc
            except Exception as e:
                last_err = e
                continue
    raise last_err

def build_label_space_and_multihot(
    col_r1: pd.Series, col_r2: pd.Series
) -> Tuple[List[str], np.ndarray, np.ndarray, List[Set[str]], List[Set[str]]]:
    """将 R1/R2 解析为多标签集合并转为 multi-hot 矩阵。"""
    r1_sets = [tokens_from_cell(v) for v in col_r1]
    r2_sets = [tokens_from_cell(v) for v in col_r2]
    all_tags = sorted(set().union(*r1_sets, *r2_sets), key=lambda x: x.lower())
    tag_index = {t: i for i, t in enumerate(all_tags)}
    Y1 = np.zeros((len(r1_sets), len(all_tags)), dtype=int)
    Y2 = np.zeros((len(r2_sets), len(all_tags)), dtype=int)
    for i, (s1, s2) in enumerate(zip(r1_sets, r2_sets)):
        for t in s1:
            Y1[i, tag_index[t]] = 1
        for t in s2:
            Y2[i, tag_index[t]] = 1
    return all_tags, Y1, Y2, r1_sets, r2_sets

def subset_accuracy(r1_sets: List[Set[str]], r2_sets: List[Set[str]]) -> float:
    eq = [1.0 if a == b else 0.0 for a, b in zip(r1_sets, r2_sets)]
    return float(np.mean(eq)) if len(eq) else np.nan

def average_jaccard(r1_sets: List[Set[str]], r2_sets: List[Set[str]]) -> float:
    def jacc(a, b):
        if len(a) == 0 and len(b) == 0:
            return 1.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union else 1.0
    scores = [jacc(a, b) for a, b in zip(r1_sets, r2_sets)]
    return float(np.mean(scores)) if len(scores) else np.nan

# -----------------------
# 主程序
# -----------------------
def main():
    # 自动检测 CSV
    csv_path = os.path.join(os.getcwd(), CSV_NAME)
    if not os.path.exists(csv_path):
        print(f"[ERROR] 找不到 {CSV_NAME}，请确保该文件与脚本在同一目录。", file=sys.stderr)
        sys.exit(1)

    ensure_outdir(OUTPUT_DIR)

    # 稳健读取
    try:
        df, used_encoding = read_csv_safely(csv_path)
        print(f"[INFO] CSV 读取成功，使用编码：{used_encoding}")
    except Exception as e:
        print(f"[ERROR] CSV 打开失败：{e}", file=sys.stderr)
        print("       建议：用 Excel 另存为 UTF-8 CSV，或尝试 gb18030/utf-8-sig/cp936/latin1 编码。", file=sys.stderr)
        sys.exit(1)

    # 必要列检查
    required_cols = [f"{d}_R1" for d in DIMENSIONS] + [f"{d}_R2" for d in DIMENSIONS]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[ERROR] 缺少列: {missing}", file=sys.stderr)
        sys.exit(1)

    # 结果容器
    results = []
    micro_f1_map = {}

    # 逐维度计算多标签指标（以 Micro F1 为主）
    for dim in DIMENSIONS:
        c1, c2 = f"{dim}_R1", f"{dim}_R2"
        all_tags, Y1, Y2, r1_sets, r2_sets = build_label_space_and_multihot(df[c1], df[c2])

        micro_f1 = f1_score(Y1, Y2, average="micro", zero_division=0)
        macro_f1 = f1_score(Y1, Y2, average="macro", zero_division=0)
        sub_acc   = subset_accuracy(r1_sets, r2_sets)
        avg_jac   = average_jaccard(r1_sets, r2_sets)
        ham_score = 1.0 - hamming_loss(Y1, Y2)

        micro_f1_map[dim] = float(micro_f1)

        results.append({
            "dimension": dim,
            "n_items": int(len(df)),
            "n_tags": int(len(all_tags)),
            "micro_f1": round(float(micro_f1), 4),   # —— 主结果
            "macro_f1": round(float(macro_f1), 4),
            "subset_accuracy": round(float(sub_acc), 4),
            "avg_jaccard": round(float(avg_jac), 4),
            "hamming_score": round(float(ham_score), 4),
        })

    # 导出 CSV（Micro F1 靠前）
    metrics_df = pd.DataFrame(results)
    col_order = ["dimension", "n_items", "n_tags", "micro_f1", "macro_f1",
                 "subset_accuracy", "avg_jaccard", "hamming_score"]
    metrics_df = metrics_df[col_order]
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "metrics_per_dimension.csv"), index=False, encoding="utf-8")
    print(f"[OK] 指标已导出: {os.path.join(OUTPUT_DIR, 'metrics_per_dimension.csv')}")

    # -------------------
    # 绘制 Micro F1 水平渐变条形图（与 Challenges 图一致的设计语言）
    # -------------------
    dims = DIMENSIONS
    vals = np.array([micro_f1_map[d] for d in dims])
    max_val = float(vals.max()) if len(vals) else 1.0

    fig, ax = plt.subplots(figsize=(12, 6))

    # 渐变色（按值做归一化）
    # 为避免过浅，给一个下限：0.35
    norm_vals = 0.35 + 0.65 * (vals / max_val if max_val > 0 else vals)
    colors = plt.cm.Blues(norm_vals)

    # 水平条形
    bars = ax.barh(
        dims,
        vals,
        color=colors,
        edgecolor='black',
        linewidth=0.7,
        height=0.7
    )

    # 右侧标注（两位小数）
    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + 0.03,                                   # 与 x 轴比例相关（F1 ∈ [0,1]）
            bar.get_y() + bar.get_height() / 2,
            f'{w:.2f}',
            va='center',
            ha='left',
            fontsize=11
        )

    # 轴与标题
    ax.set_xlim(0, 1.0)
    ax.set_xlabel('Micro F1', labelpad=8, fontsize=12)
    ax.set_ylabel('Dimension', labelpad=8, fontsize=12)
    ax.set_title('Multi-label Micro F1 per Dimension', pad=10, fontsize=14, weight='bold')

    # 去除顶部和右边框，添加虚线网格
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.grid(True, linestyle='--', alpha=0.3)

    # 从高到低顺序显示（与参考图一致）
    ax.invert_yaxis()

    # Y 轴刻度字体
    ax.tick_params(axis='y', labelsize=13)

    # 自动左边距，避免长标签被裁剪
    max_label_len = max(len(str(lab)) for lab in dims) if len(dims) else 10
    left_margin = 0.15 + max_label_len * 0.015
    left_margin = min(max(left_margin, 0.15), 0.4)

    fig.subplots_adjust(left=left_margin, right=0.95, top=0.9, bottom=0.12)

    # 高分辨率保存
    out_path = os.path.join(OUTPUT_DIR, "micro_f1_bar.png")
    plt.savefig(out_path, dpi=600, bbox_inches='tight', transparent=False)
    plt.close(fig)

    print(f"[OK] 图已保存至：{out_path}")
    print(f"[DONE] 分析完成（Micro F1 为主结果）。输出目录：{OUTPUT_DIR}")

if __name__ == "__main__":
    main()
