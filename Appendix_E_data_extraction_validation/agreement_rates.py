import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import os

# ---------- Global font and style configuration ----------
plt.rcParams['font.family'] = 'Times New Roman'

def generate_agreement_rates_plot():
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Load the field agreement data
    field_agreement = pd.read_csv("results/field_agreement_rates.csv")

    # Sort by agreement rate for better visualization
    field_agreement = field_agreement.sort_values('AgreementRate', ascending=True)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create color gradient based on agreement rate
    colors = plt.cm.Blues(field_agreement['AgreementRate'])

    # Draw horizontal bars
    bars = ax.barh(
        field_agreement['Field'],
        field_agreement['AgreementRate'],
        color=colors,
        edgecolor='black',
        linewidth=0.7,
        height=0.7
    )

    # Add value annotations
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.03,  # Slightly offset from bar
            bar.get_y() + bar.get_height() / 2,
            f'{width:.0%}',
            va='center',
            ha='left',
            fontsize=11
        )

    # Set axis labels and title
    ax.set_xlabel('Agreement Rate', labelpad=8, fontsize=12)
    ax.set_ylabel('Field', labelpad=8, fontsize=12)
    ax.set_title('Field-wise Agreement Rates Between AI Extraction and Human Validation',
                pad=10, fontsize=14, weight='bold')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add light grid lines (x-axis only)
    ax.xaxis.grid(True, linestyle='--', alpha=0.3)

    # Set x-axis limits and ticks
    ax.set_xlim(0, 1.1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])

    # Adjust layout to prevent label clipping
    max_label_len = max(len(str(label)) for label in field_agreement['Field'])
    left_margin = 0.15 + max_label_len * 0.015
    left_margin = min(max(left_margin, 0.15), 0.4)  # Limit range

    fig.subplots_adjust(left=left_margin, right=0.95, top=0.9, bottom=0.15)

    # Save high-quality figure
    plot_path = os.path.join("results", "agreement_rates.png")
    plt.savefig(
        plot_path,
        dpi=600,
        bbox_inches='tight',
        transparent=False
    )
    plt.close()

    print(f"Agreement rates plot saved to: {plot_path}")

if __name__ == "__main__":
    generate_agreement_rates_plot()