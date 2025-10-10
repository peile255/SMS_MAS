import bibtexparser
import pandas as pd

# 读取 bib 文件
with open('2.bib', 'r', encoding='utf-8') as bib_file:
    bib_database = bibtexparser.load(bib_file)

# 提取 title、abstract 和 keywords
records = []
for entry in bib_database.entries:
    title = entry.get('title', '').strip()
    abstract = entry.get('abstract', '').strip()
    keywords = entry.get('keywords', '').strip()

    # 跳过缺少必要字段的记录
    if title and abstract and keywords:
        records.append({
            'title': title.replace('\n', ' ').replace('"', "'"),
            'abstract': abstract.replace('\n', ' ').replace('"', "'"),
            'keywords': keywords.replace('\n', ' ').replace('"', "'")
        })

# 保存为 CSV
if records:
    df = pd.DataFrame(records)
    df.to_csv('bib2_output.csv', index=False)
    print("✅ 提取成功，已生成 'bib2_output.csv'")
else:
    print("⚠️ 没有找到完整的条目（包含 title、abstract 和 keywords）")
