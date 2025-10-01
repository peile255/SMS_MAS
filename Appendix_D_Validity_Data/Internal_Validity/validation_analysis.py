import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
    cohen_kappa_score,
)
from sklearn.utils import resample
from statsmodels.stats.contingency_tables import mcnemar

# ---------------------------
# Constants
# ---------------------------
RELEVANT = "Relevant"
IRRELEVANT = "Irrelevant"
VALID_LABELS = {RELEVANT, IRRELEVANT}

# ---------------------------
# IO & Preprocess
# ---------------------------
def load_and_preprocess(filepath, r1_col="R1", r2_col="R2"):
    """Load CSV, clean labels, and compute conflict flag."""
    df = pd.read_csv(filepath)

    # Ensure required columns exist
    required = ["Article_Number", "Title", r1_col, r2_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}. Required: {required}")

    # Coerce to string (in case there are NaNs) and strip
    df[r1_col] = df[r1_col].astype(str).str.strip()
    df[r2_col] = df[r2_col].astype(str).str.strip()

    # Normalize common variants (case-insensitive)
    def norm_label(x):
        x_low = x.lower()
        if x_low in ["relevant", "rel", "r", "1", "yes", "y"]:
            return RELEVANT
        if x_low in ["irrelevant", "irrel", "not relevant", "nr", "0", "no", "n"]:
            return IRRELEVANT
        return x  # keep original for later checking

    df[r1_col] = df[r1_col].map(norm_label)
    df[r2_col] = df[r2_col].map(norm_label)

    # Warn if unseen labels exist
    unseen_r1 = sorted(set(df[r1_col].unique()) - VALID_LABELS)
    unseen_r2 = sorted(set(df[r2_col].unique()) - VALID_LABELS)
    if unseen_r1 or unseen_r2:
        print(
            "⚠️ Found unexpected labels.\n"
            f"  {r1_col} unexpected: {unseen_r1}\n"
            f"  {r2_col} unexpected: {unseen_r2}\n"
            "Only 'Relevant' / 'Irrelevant' are evaluated. Others will remain and may affect metrics."
        )

    # Conflict flag
    df["Conflict"] = df[r1_col] != df[r2_col]
    return df.rename(columns={r1_col: "R1", r2_col: "R2"})

# ---------------------------
# Metrics
# ---------------------------
def safe_metrics(y_true, y_pred):
    """Safely compute precision, recall, f1 with zero_division handled."""
    try:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
    except Exception:
        precision, recall, f1 = 0.0, 0.0, 0.0
    return precision, recall, f1


def calculate_metrics(df, n_boot=1000, random_state=42):
    """
    Compute evaluation metrics treating R1 as the reference and R2 as the comparator.
    (If you want fully symmetric reporting, rely on Kappa, conflict rate, and McNemar.)
    """
    y_true = (df["R1"] == RELEVANT).astype(int)
    y_pred = (df["R2"] == RELEVANT).astype(int)

    precision, recall, f1 = safe_metrics(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    # Confusion matrix with semantic labels (row: R1, col: R2)
    cm = confusion_matrix(df["R1"], df["R2"], labels=[RELEVANT, IRRELEVANT])

    # Bootstrap CIs
    rng = np.random.default_rng(random_state)
    boot_vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(df), len(df))
        sample = df.iloc[idx]

        y_true_b = (sample["R1"] == RELEVANT).astype(int)
        y_pred_b = (sample["R2"] == RELEVANT).astype(int)

        # Skip degenerate all-same cases to avoid nan in metrics
        if y_true_b.nunique() == 1 and y_pred_b.nunique() == 1:
            continue

        acc_b = accuracy_score(y_true_b, y_pred_b)
        prec_b, rec_b, _ = safe_metrics(y_true_b, y_pred_b)
        boot_vals.append([acc_b, prec_b, rec_b])

    boot_vals = np.array(boot_vals) if len(boot_vals) else np.array([[accuracy, precision, recall]])
    ci_accuracy = np.percentile(boot_vals[:, 0], [2.5, 97.5])
    ci_precision = np.percentile(boot_vals[:, 1], [2.5, 97.5])
    ci_recall = np.percentile(boot_vals[:, 2], [2.5, 97.5])

    # Extra symmetric stats
    conflict_rate = df["Conflict"].mean()
    r1_only = ((df["R1"] == RELEVANT) & (df["R2"] == IRRELEVANT)).sum()
    r2_only = ((df["R2"] == RELEVANT) & (df["R1"] == IRRELEVANT)).sum()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "kappa": kappa,
        "confusion_matrix": cm,
        "ci_accuracy": ci_accuracy,
        "ci_precision": ci_precision,
        "ci_recall": ci_recall,
        "boot_metrics": boot_vals,
        "conflict_rate": conflict_rate,
        "r1_only_count": int(r1_only),
        "r2_only_count": int(r2_only),
    }


def statistical_tests(df):
    """
    McNemar test on paired binary labels:
    n11: R1=Rel, R2=Rel
    n10: R1=Rel, R2=Irr
    n01: R1=Irr, R2=Rel
    n00: R1=Irr, R2=Irr
    table = [[n00, n01],
             [n10, n11]]
    """
    r1_rel = df["R1"] == RELEVANT
    r2_rel = df["R2"] == RELEVANT

    n11 = np.sum(r1_rel & r2_rel)
    n10 = np.sum(r1_rel & (~r2_rel))
    n01 = np.sum((~r1_rel) & r2_rel)
    n00 = np.sum((~r1_rel) & (~r2_rel))

    table = np.array([[n00, n01], [n10, n11]])
    mcnemar_p = mcnemar(table, exact=True).pvalue
    return {"mcnemar_p": mcnemar_p, "mcnemar_table": table}

# ---------------------------
# Plots
# ---------------------------
def plot_confusion_matrix(cm, filename):
    """Plot heatmap of the 2x2 matrix with semantic tick labels."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[RELEVANT, IRRELEVANT],
        yticklabels=[RELEVANT, IRRELEVANT],
    )
    plt.xlabel("R2")
    plt.ylabel("R1")
    plt.title("Confusion Matrix (R1 vs R2)")
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{filename}.png", bbox_inches="tight", dpi=300)
    plt.close()


def plot_metric_distributions(boot_metrics, filename):
    """Plot bootstrap distributions for Accuracy / Precision / Recall."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    sns.histplot(boot_metrics[:, 0], ax=axes[0], kde=True)
    axes[0].set_title("Accuracy Distribution")
    axes[0].set_xlim(0.0, 1.0)

    sns.histplot(boot_metrics[:, 1], ax=axes[1], kde=True)
    axes[1].set_title("Precision Distribution")
    axes[1].set_xlim(0.0, 1.0)

    sns.histplot(boot_metrics[:, 2], ax=axes[2], kde=True)
    axes[2].set_title("Recall Distribution")
    axes[2].set_xlim(0.0, 1.0)

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{filename}.png", bbox_inches="tight", dpi=300)
    plt.close()

# ---------------------------
# Conflict / Error Listing
# ---------------------------
def save_conflict_analysis(df, filename):
    """Save disagreements between R1 and R2 (author-to-author)."""
    r1_only = df[(df["R1"] == RELEVANT) & (df["R2"] == IRRELEVANT)]
    r2_only = df[(df["R2"] == RELEVANT) & (df["R1"] == IRRELEVANT)]

    os.makedirs("results", exist_ok=True)
    with open(f"results/{filename}.txt", "w", encoding="utf-8") as f:
        f.write("=== R1 marked Relevant, R2 marked Irrelevant ===\n")
        for _, row in r1_only.iterrows():
            f.write(f"ID {int(row['Article_Number'])}: {row['Title']}\n")

        f.write("\n=== R2 marked Relevant, R1 marked Irrelevant ===\n")
        for _, row in r2_only.iterrows():
            f.write(f"ID {int(row['Article_Number'])}: {row['Title']}\n")


def save_error_analysis(df, filename):
    """Backward-compatible alias keeping old function name."""
    return save_conflict_analysis(df, filename)

# ---------------------------
# Save Results
# ---------------------------
def save_results(metrics, stats, filename):
    """Save overall metrics, CIs, conflict stats, and McNemar p to CSV."""
    os.makedirs("results", exist_ok=True)

    results_df = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "Kappa",
                "ConflictRate",
                "R1_only(Rel&!R2)",
                "R2_only(Rel&!R1)",
                "McNemar_p",
            ],
            "Value": [
                metrics["accuracy"],
                metrics["precision"],
                metrics["recall"],
                metrics["f1"],
                metrics["kappa"],
                metrics["conflict_rate"],
                metrics["r1_only_count"],
                metrics["r2_only_count"],
                stats["mcnemar_p"],
            ],
            "CI_Lower": [
                metrics["ci_accuracy"][0],
                metrics["ci_precision"][0],
                metrics["ci_recall"][0],
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
            "CI_Upper": [
                metrics["ci_accuracy"][1],
                metrics["ci_precision"][1],
                metrics["ci_recall"][1],
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
        }
    )
    results_df.to_csv(f"results/{filename}.csv", index=False)

# ---------------------------
# CLI & Main
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Validate R1 vs R2 screening agreement.")
    p.add_argument(
        "-i", "--input", default="screen_process.csv", help="Input CSV path (default: screen_process.csv)"
        )
    p.add_argument(
        "--r1_col", default="R1", help="Column name for reviewer 1 (default: R1)"
        )
    p.add_argument(
        "--r2_col", default="R2", help="Column name for reviewer 2 (default: R2)"
        )
    p.add_argument(
        "--n_boot", type=int, default=1000, help="Bootstrap iterations (default: 1000)"
        )
    p.add_argument(
        "--seed", type=int, default=42, help="Random seed for bootstrap (default: 42)"
        )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs("results", exist_ok=True)

    # Step 1: Load data
    df = load_and_preprocess(args.input, r1_col=args.r1_col, r2_col=args.r2_col)

    # Step 2: Compute evaluation metrics
    metrics = calculate_metrics(df, n_boot=args.n_boot, random_state=args.seed)

    # Step 3: Statistical significance testing (symmetric, paired)
    stats = statistical_tests(df)

    # Step 4: Save results and plots
    plot_confusion_matrix(metrics["confusion_matrix"], "confusion_matrix")
    plot_metric_distributions(metrics["boot_metrics"], "metric_distributions")
    save_conflict_analysis(df, "conflict_analysis")  # or save_error_analysis for backward compat
    save_results(metrics, stats, "validation_metrics")

    # Step 5: Print summary
    total = len(df)
    n_rel_r1 = int((df["R1"] == RELEVANT).sum())
    print(
        f"""
=========== Screening Validation Results ===========
Total samples: {total} (R1: Relevant={n_rel_r1}, Irrelevant={total - n_rel_r1})
----------------------------------------------------
Accuracy : {metrics['accuracy']:.1%}  (95% CI: {metrics['ci_accuracy'][0]:.1%}-{metrics['ci_accuracy'][1]:.1%})
Precision: {metrics['precision']:.1%} (95% CI: {metrics['ci_precision'][0]:.1%}-{metrics['ci_precision'][1]:.1%})
Recall   : {metrics['recall']:.1%} (95% CI: {metrics['ci_recall'][0]:.1%}-{metrics['ci_recall'][1]:.1%})
F1 Score : {metrics['f1']:.1%}
Kappa    : {metrics['kappa']:.3f}
----------------------------------------------------
Conflict Rate: {metrics['conflict_rate']:.1%}
R1-only Relevant (& R2 Irrelevant): {metrics['r1_only_count']}
R2-only Relevant (& R1 Irrelevant): {metrics['r2_only_count']}
McNemar Test p-value: {stats['mcnemar_p']:.4f} ({'not significant' if stats['mcnemar_p'] > 0.05 else 'significant'})
====================================================
"""
    )


if __name__ == "__main__":
    main()
