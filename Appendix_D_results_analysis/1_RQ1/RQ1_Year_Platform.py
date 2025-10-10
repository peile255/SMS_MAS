import pandas as pd
import matplotlib.pyplot as plt
import os
import ast

# 1. 创建 figures 文件夹（如果不存在）
os.makedirs('figures', exist_ok=True)

# 2. 读取 CSV 文件
df = pd.read_csv("Articles.csv")

# 3. 将字符串形式的列表字段转换为真正的列表对象
df['Platform_Type'] = df['Platform_Type'].apply(ast.literal_eval)
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')  # 确保 Year 为数值型

# 4. 拆分 Platform_Type 多标签为多行
df_exploded = df.explode('Platform_Type')

# ✅ 去除空白、None、nan 平台类型
df_exploded['Platform_Type'] = df_exploded['Platform_Type'].astype(str).str.strip()
df_exploded = df_exploded[df_exploded['Platform_Type'].notna()]
df_exploded = df_exploded[df_exploded['Platform_Type'].str.lower() != 'nan']
df_exploded = df_exploded[df_exploded['Platform_Type'] != '']

# 5. 统计每年每种平台类型的数量
platform_trends = df_exploded.groupby(['Year', 'Platform_Type']).size().unstack(fill_value=0)

# ✅ 再次确保没有 'nan' 列
if 'nan' in platform_trends.columns:
    platform_trends = platform_trends.drop(columns='nan')

# ✅ 保存统计数据为 CSV
platform_trends.to_csv('figures/RQ1_Year_Platform.csv')

# 6. 画图
fig, ax = plt.subplots(figsize=(10, 6))
platform_trends.plot(kind='line', marker='o', ax=ax)

# 设置标题和标签
ax.set_title('Platform Type Trends Over Time')
ax.set_xlabel('Year')
ax.set_ylabel('Number of Articles')
ax.grid(True)

# 调整图例位置，避免遮挡
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
ax.legend(title='Platform Type', loc='center left', bbox_to_anchor=(1.0, 0.5))

# 7. 保存图像
plt.savefig('figures/RQ1_Year_Platform.png', bbox_inches='tight', dpi=300)
plt.close()
