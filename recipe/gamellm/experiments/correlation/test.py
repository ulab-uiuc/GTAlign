import json
import pandas as pd

# 读取 JSON 文件
data = []
with open("/mnt/bn/heheda/verl/0910_medium_cobb/100.jsonl", "r") as f:
    for line in f:
        data.append(json.loads(line.strip()))

# 转换成 DataFrame
df = pd.DataFrame(data)

# 取出需要的列
cols = ["utility", "llm_reward", "user_reward", "answer_reward"]
df = df[cols]

# 计算相关系数矩阵（皮尔逊相关系数）
pearson_corr = df.corr(method="pearson")

# 计算相关系数矩阵（Kendall Tau）
kendall_corr = df.corr(method="kendall")

# 计算相关系数矩阵（Spearman 秩相关系数）
spearman_corr = df.corr(method="spearman")

print("皮尔逊相关系数矩阵：")
print(pearson_corr)

print("\nKendall Tau 相关系数矩阵：")
print(kendall_corr)

print("\nSpearman 秩相关系数矩阵：")
print(spearman_corr)
