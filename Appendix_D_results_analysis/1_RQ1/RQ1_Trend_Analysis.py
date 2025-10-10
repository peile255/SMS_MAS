import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

def analyze_publication_trends(input_csv="Articles.csv", output_dir="figures"):
    """
    Analyze publication trends over time with IEEE-compliant font styles,
    unified color palette with pie chart, and export data and visualization.
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 1. Load and prepare data
        df = pd.read_csv(input_csv)

        if 'Year' not in df.columns:
            raise ValueError("CSV file does not contain 'Year' column")

        df = df.dropna(subset=['Year'])
        df['Year'] = df['Year'].astype(int)
        current_year = datetime.now().year
        df = df[(df['Year'] >= 2015) & (df['Year'] <= current_year)]

        # 2. Calculate yearly publication counts
        year_counts = df['Year'].value_counts().sort_index()
        all_years = pd.Series(index=np.arange(2015, current_year + 1), dtype='float64')
        publications = all_years.combine(year_counts, lambda x, y: y if not np.isnan(y) else 0).astype(int)

        # 3. Save the data to CSV
        data_df = pd.DataFrame({
            'Year': publications.index,
            'Publications': publications.values
        })
        data_csv_path = os.path.join(output_dir, 'RQ1_1_trend_analysis.csv')
        data_df.to_csv(data_csv_path, index=False)
        print(f"Publication trend data saved to: {data_csv_path}")

        # 4. Define visual phases
        phases = [
            {"label": "Emerging Phase", "start": 2015, "end": 2020, "color": "#BFDDF5"},   # 浅蓝
            {"label": "Growth Phase", "start": 2021, "end": 2022, "color": "#9DC3E6"},     # 蓝绿
            {"label": "Boom Phase", "start": 2023, "end": current_year, "color": "#6699CC"}  # 中蓝
        ]

        # 5. Global font config
        plt.rcParams['font.family'] = 'Times New Roman'

        # 6. Visualization
        plt.figure(figsize=(6, 4), dpi=600)
        plt.rcParams['axes.facecolor'] = '#F5F5F5'
        plt.gca().set_facecolor('#F5F5F5')

        # Main line color (unified with pie chart)
        line_color = '#4A90E2'

        plt.plot(publications.index, publications.values,
                 marker='o', markersize=8, linewidth=2.5,
                 color=line_color, markerfacecolor='white',
                 markeredgewidth=2, markeredgecolor=line_color,
                 label='Publications', zorder=3)

        # Phase background shading
        for phase in phases:
            plt.axvspan(phase["start"] - 0.5, phase["end"] + 0.5,
                        facecolor=phase["color"], alpha=0.4, zorder=1)
            mid_x = (phase["start"] + phase["end"]) / 2
            plt.text(mid_x, plt.ylim()[1] * 1.2,
                     f"{phase['label']}\n({phase['start']}-{phase['end']})",
                     ha='center', va='center', fontsize=11,
                     bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray', boxstyle='round,pad=0.3'),
                     zorder=2)

        # Labels and title
        plt.title(f'Annual Publications Trend (2015–{current_year})',
                  fontsize=14, pad=38, fontweight='bold', color='#333333')
        plt.xlabel('Year', fontsize=12, labelpad=10, color='#333333')
        plt.ylabel('Number of Publications', fontsize=12, labelpad=10, color='#333333')

        plt.xticks(publications.index, rotation=45, fontsize=11)
        plt.yticks(fontsize=11)

        # Axis and grid
        plt.grid(True, linestyle='-', alpha=0.3, color='gray', zorder=0)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_color('#666666')
        plt.gca().spines['bottom'].set_color('#666666')

        # Data point labels
        for year, count in zip(publications.index, publications.values):
            plt.text(year, count + 2, str(count),
                     ha='center', va='bottom', fontsize=11,
                     color=line_color, fontweight='bold')

        plt.ylim(0, publications.max() * 1.15)

        # Footer note
        plt.figtext(0.99, 0.01,
                    "Note: 2025 data includes publications only through May.",
                    wrap=True, horizontalalignment='right',
                    fontsize=10, color='#666666')

        # Layout and save
        plt.tight_layout(pad=2.0)
        image_path = os.path.join(output_dir, 'RQ1_1_trend_analysis.png')
        plt.savefig(image_path, dpi=600, bbox_inches='tight', transparent=False)
        plt.close()

        print(f"Trend analysis chart saved to: {image_path}")
        print(f"Total publications analyzed: {len(df)}")
        print(f"Time span covered: {publications.index.min()}–{publications.index.max()}")

    except FileNotFoundError:
        print(f"Error: File '{input_csv}' not found.")
    except ValueError as ve:
        print(f"Value error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    analyze_publication_trends()
