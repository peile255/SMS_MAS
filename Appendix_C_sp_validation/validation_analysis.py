import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                             accuracy_score, cohen_kappa_score)
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.utils import resample
from scipy import stats

# 创建结果文件夹
os.makedirs('results', exist_ok=True)


def load_and_preprocess(filepath):
    """加载并预处理数据"""
    df = pd.read_csv(filepath)

    # 清洗标签格式
    df['Manual_Label'] = df['Manual_Label'].str.strip()
    df['AI_Prediction'] = df['AI_Prediction'].str.strip()

    # 标记冲突样本
    df['Conflict'] = df['Manual_Label'] != df['AI_Prediction']
    return df


def safe_metrics(y_true, y_pred):
    """安全计算指标，处理零除情况"""
    try:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred,
            average='binary',
            zero_division=0  # 处理除以零的情况
        )
    except:
        precision, recall, f1 = 0.0, 0.0, 0.0
    return precision, recall, f1


def calculate_metrics(df):
    """计算核心指标"""
    # 确保标签是二进制数值（0和1）
    y_true = (df['Manual_Label'] == 'Relevant').astype(int)
    y_pred = (df['AI_Prediction'] == 'Relevant').astype(int)

    # 基础指标
    precision, recall, f1 = safe_metrics(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    # 混淆矩阵（使用原始标签）
    cm = confusion_matrix(df['Manual_Label'], df['AI_Prediction'],
                          labels=['Relevant', 'Irrelevant'])

    # Bootstrap置信区间（带错误处理）
    boot_metrics = []
    for _ in range(1000):
        sample = resample(df, replace=True)
        y_true_boot = (sample['Manual_Label'] == 'Relevant').astype(int)
        y_pred_boot = (sample['AI_Prediction'] == 'Relevant').astype(int)

        # 跳过全0样本
        if sum(y_true_boot) == 0 and sum(y_pred_boot) == 0:
            continue

        acc = accuracy_score(y_true_boot, y_pred_boot)
        prec, rec, f1_boot = safe_metrics(y_true_boot, y_pred_boot)
        boot_metrics.append([acc, prec, rec])

    boot_metrics = np.array(boot_metrics)
    ci_accuracy = np.percentile(boot_metrics[:, 0], [2.5, 97.5])
    ci_precision = np.percentile(boot_metrics[:, 1], [2.5, 97.5])
    ci_recall = np.percentile(boot_metrics[:, 2], [2.5, 97.5])

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'kappa': kappa,
        'confusion_matrix': cm,
        'ci_accuracy': ci_accuracy,
        'ci_precision': ci_precision,
        'ci_recall': ci_recall,
        'boot_metrics': boot_metrics
    }


def statistical_tests(df):
    """执行统计检验"""
    # McNemar检验
    cm = confusion_matrix(df['Manual_Label'], df['AI_Prediction'],
                          labels=['Relevant', 'Irrelevant'])
    mcnemar_table = [[cm[1, 1], cm[0, 1]], [cm[1, 0], cm[0, 0]]]  # TN, FP, FN, TP
    mcnemar_p = mcnemar(mcnemar_table, exact=True).pvalue

    return {
        'mcnemar_p': mcnemar_p
    }


def plot_confusion_matrix(cm, filename):
    """绘制混淆矩阵"""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Relevant', 'Irrelevant'],
                yticklabels=['Relevant', 'Irrelevant'])
    plt.xlabel('AI Prediction')
    plt.ylabel('Manual Label')
    plt.title('Confusion Matrix')
    plt.savefig(f'results/{filename}.png', bbox_inches='tight', dpi=300)
    plt.close()


def plot_metric_distributions(boot_metrics, filename):
    """绘制指标分布"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Accuracy
    sns.histplot(boot_metrics[:, 0], ax=axes[0], kde=True)
    axes[0].set_title('Accuracy Distribution')
    axes[0].set_xlim(0.5, 1.0)

    # Precision
    sns.histplot(boot_metrics[:, 1], ax=axes[1], kde=True)
    axes[1].set_title('Precision Distribution')
    axes[1].set_xlim(0.0, 1.0)

    # Recall
    sns.histplot(boot_metrics[:, 2], ax=axes[2], kde=True)
    axes[2].set_title('Recall Distribution')
    axes[2].set_xlim(0.0, 1.0)

    plt.tight_layout()
    plt.savefig(f'results/{filename}.png', bbox_inches='tight', dpi=300)
    plt.close()


def save_error_analysis(df, filename):
    """保存错误分析结果"""
    # 假阳性
    fp = df[(df['AI_Prediction'] == 'Relevant') &
            (df['Manual_Label'] == 'Irrelevant')]

    # 假阴性
    fn = df[(df['AI_Prediction'] == 'Irrelevant') &
            (df['Manual_Label'] == 'Relevant')]

    with open(f'results/{filename}.txt', 'w') as f:
        f.write("=== False Positives (AI错误标记为相关) ===\n")
        for _, row in fp.iterrows():
            f.write(f"ID {int(row['Article_Number'])}: {row['Title']}\n")

        f.write("\n=== False Negatives (AI漏标的相关文献) ===\n")
        for _, row in fn.iterrows():
            f.write(f"ID {int(row['Article_Number'])}: {row['Title']}\n")


def save_results(metrics, stats, filename):
    """保存结果到CSV"""
    results_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1', 'Kappa', 'McNemar_p'],
        'Value': [
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1'],
            metrics['kappa'],
            stats['mcnemar_p']
        ],
        'CI_Lower': [
            metrics['ci_accuracy'][0],
            metrics['ci_precision'][0],
            metrics['ci_recall'][0],
            np.nan,
            np.nan,
            np.nan
        ],
        'CI_Upper': [
            metrics['ci_accuracy'][1],
            metrics['ci_precision'][1],
            metrics['ci_recall'][1],
            np.nan,
            np.nan,
            np.nan
        ]
    })
    results_df.to_csv(f'results/{filename}.csv', index=False)


def main():
    # 1. 加载数据
    df = load_and_preprocess('screen_process_extracted.csv')

    # 2. 计算指标
    metrics = calculate_metrics(df)

    # 3. 统计检验
    stats = statistical_tests(df)

    # 4. 可视化与保存
    plot_confusion_matrix(metrics['confusion_matrix'], 'confusion_matrix')
    plot_metric_distributions(metrics['boot_metrics'], 'metric_distributions')
    save_error_analysis(df, 'error_analysis')
    save_results(metrics, stats, 'validation_metrics')

    # 5. 打印关键结果
    print(f"""
    ============ AI辅助筛选验证结果 ============
    总样本数: {len(df)}篇 (相关: {sum(df['Manual_Label'] == 'Relevant')}篇)
    -----------------------------------------
    准确率: {metrics['accuracy']:.1%} (95% CI: {metrics['ci_accuracy'][0]:.1%}-{metrics['ci_accuracy'][1]:.1%})
    精确率: {metrics['precision']:.1%} (CI: {metrics['ci_precision'][0]:.1%}-{metrics['ci_precision'][1]:.1%})
    召回率: {metrics['recall']:.1%} (CI: {metrics['ci_recall'][0]:.1%}-{metrics['ci_recall'][1]:.1%})
    F1分数: {metrics['f1']:.1%}
    -----------------------------------------
    Kappa一致性: {metrics['kappa']:.3f} (中等强度)
    McNemar检验p值: {stats['mcnemar_p']:.4f} ({"不显著" if stats['mcnemar_p'] > 0.05 else "显著"})
    ===========================================
    """)


if __name__ == '__main__':
    main()