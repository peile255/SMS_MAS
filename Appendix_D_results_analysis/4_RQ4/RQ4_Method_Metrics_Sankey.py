import os
import pandas as pd
import matplotlib.pyplot as plt
import ast
from pySankey.sankey import sankey

# ----------------------------
# Setup & Styling
# ----------------------------

output_dir = "figures"
os.makedirs(output_dir, exist_ok=True)

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# ----------------------------
# Load & Clean Data
# ----------------------------

df = pd.read_csv('Articles.csv')

# Drop rows with missing values
df = df.dropna(subset=['Evaluation_Method', 'Metrics'])

# 安全解析字符串为列表
def safe_literal_eval(val):
    try:
        if isinstance(val, str) and val.strip().startswith('[') and val.strip().endswith(']'):
            return ast.literal_eval(val)
        return val
    except Exception:
        return None

df['Evaluation_Method'] = df['Evaluation_Method'].apply(safe_literal_eval)
df['Metrics'] = df['Metrics'].apply(safe_literal_eval)

df = df.dropna(subset=['Evaluation_Method', 'Metrics'])

# ----------------------------
# Explode for Sankey Mapping
# ----------------------------

if isinstance(df['Evaluation_Method'].iloc[0], list):
    df = df.explode('Evaluation_Method')
if isinstance(df['Metrics'].iloc[0], list):
    df = df.explode('Metrics')

df = df.dropna(subset=['Evaluation_Method', 'Metrics'])

# ----------------------------
# Sankey Data and Plot
# ----------------------------

sankey_df = df[['Evaluation_Method', 'Metrics']].copy()
sankey_df.columns = ['Source', 'Target']

# Save data
sankey_df.to_csv(os.path.join(output_dir, 'RQ4_Method_Metrics_Sankey.csv'), index=False)

# Plot
plt.figure(figsize=(16, 10))
sankey(
    left=sankey_df['Source'],
    right=sankey_df['Target'],
    aspect=20,
    fontsize=10
)

plt.title("Mapping Between Evaluation Methods and Metrics",
          fontsize=12, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'RQ4_Method_Metrics_Sankey.png'),
            dpi=300, bbox_inches='tight')
plt.close()

print("✔ Sankey diagram saved to 'figures/RQ4_Method_Metrics_Sankey.png'")
print("✔ Sankey data saved to 'figures/RQ4_Method_Metrics_Sankey.csv'")
