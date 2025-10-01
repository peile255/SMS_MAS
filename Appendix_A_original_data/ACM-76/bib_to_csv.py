import bibtexparser
import pandas as pd

# 读取 bib 文件
with open('acm.bib', 'r', encoding='utf-8') as bib_file:
    bib_database = bibtexparser.load(bib_file)

# 提取 title、abstract、keywords
records = []
for entry in bib_database.entries:
    title = entry.get('title', '').strip()
    abstract = entry.get('abstract', '').strip()

    # 有些bib文件可能用 'keywords' 或 'keyword' 表示关键词
    keywords = entry.get('keywords', '').strip()
    if not keywords:
        keywords = entry.get('keyword', '').strip()

    # 为 title 添加双引号
    if title and not (title.startswith('"') and title.endswith('"')):
        title = f'"{title}"'

    # 添加记录
    if title:
        records.append({
            'Title': title,
            'Abstract': abstract,
            'Keywords': keywords
        })

# 保存为 CSV
if records:
    df = pd.DataFrame(records)
    df.to_csv('titles.csv', index=False)
    print("Successfully extracted Title, Abstract, and Keywords to 'titles.csv'")
else:
    print("No valid entries with titles found in acm1.bib")
