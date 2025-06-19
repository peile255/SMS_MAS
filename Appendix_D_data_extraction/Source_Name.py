import pandas as pd
import re

# 加载已初步标准化的数据，假设文件名是 Standardized_Source_Name.csv
df = pd.read_csv("Standardized_Source_Name.csv")

# 特殊缩写保持大写的集合
KEEP_UPPER = {
    "IEEE", "ACM", "INFOCOM", "GLOBECOM", "SECON", "VRW", "ICC", "AAMAS",
    "MetaCom", "IoT", "CIC", "WoT", "AIS", "MAS"
}

# 映射规则：部分已知名称替换，使用不区分大小写匹配
SPECIAL_CASES = {
    r"(?i).*infocom wkshps.*": "IEEE INFOCOM Workshops",
    r"(?i).*globcom.*wkshps.*": "IEEE Global Communications Conference (GLOBECOM Workshops)",
    r"(?i).*secon.*": "IEEE International Conference on Sensing, Communication, and Networking (SECON)",
    r"(?i).*vrw.*": "IEEE Conference on Virtual Reality and 3D User Interfaces Abstracts and Workshops (VRW)",
    r"(?i).*aamas.*": "AAMAS (Autonomous Agents and Multiagent Systems)",
    r"(?i).*iot.*journal.*": "IEEE Internet of Things Journal",
    r"(?i).*iot$": "IoT (Journal)",
    r"(?i).*metacom.*": "IEEE International Conference on Metaverse Computing (MetaCom)",
    r"(?i).*cic.*iccc.*": "IEEE/CIC International Conference on Communications in China (ICCC Workshops)",
    r"(?i).*kecui.*": "Internal Technical Report - KeCui Beijing"
}

# 标题格式化，同时保持特殊缩写大写
def title_preserve_acronyms(name):
    words = name.split()
    new_words = []
    for word in words:
        # 保持大写缩写，否则首字母大写，其他小写
        if word.upper() in KEEP_UPPER:
            new_words.append(word.upper())
        else:
            # 避免全部变成小写（比如含括号或连字符）
            # 这里简化为首字母大写，剩余小写
            new_words.append(word.capitalize())
    return ' '.join(new_words)

# 应用映射规则，优先匹配特殊规则
def standardize_final(name):
    name = name.strip()
    for pattern, replacement in SPECIAL_CASES.items():
        if re.search(pattern, name):
            return replacement
    return title_preserve_acronyms(name)

# 应用处理
df["Final_Standardized_Source_Name"] = df["Standardized_Source_Name"].apply(standardize_final)

# 如果想覆盖原列
df["Source_Name"] = df["Final_Standardized_Source_Name"]

# 导出包含ID和标准化后Source_Name的新文件
df[["ID", "Source_Name"]].to_csv("Source_Name.csv", index=False)

print("✅ Source_Name.csv 已生成。")
