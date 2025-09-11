# split_by_source_and_save_parquet.py
import os
import re
from typing import Dict, List

from datasets import load_dataset
import pandas as pd

# ====== 修改以下配置 ======
PARQUET_PATH_TEST = "/mnt/bn/heheda/verl/data/gamellm_wildbreak_math_medium_abgqa/test.parquet"
PARQUET_PATH_TRAIN = "/mnt/bn/heheda/verl/data/gamellm_wildbreak_math_medium_abgqa/train.parquet"

# 输出根目录：会在其下按 source/ split 组织
OUTPUT_DIR = "/mnt/bn/heheda/verl/data/gamellm_wildbreak_math_medium_abgqa/by_source"
# =========================

def sanitize_name(s: str) -> str:
    """
    将任意 source 值转为文件系统友好的目录/文件名。
    仅保留字母数字、点、下划线、短横线，其余替换为下划线。
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_") or "unknown"

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def split_and_save(split_name: str, parquet_path: str):
    print(f"==> Loading {split_name} from: {parquet_path}")
    ds = load_dataset("parquet", data_files=parquet_path, split="train")

    if "source" not in ds.column_names:
        raise KeyError(f"`source` 字段不存在于 {split_name} 数据集中，实际列为：{ds.column_names}")

    # 获取去重后的 source 列表
    sources: List[str] = sorted(map(str, ds.unique("source")))
    print(f"[{split_name}] detected sources ({len(sources)}): {sources}")
    if len(sources) != 4:
        print(f"⚠️  警告：期望 4 类，但检测到 {len(sources)} 类。将按实际检测到的类导出。")

    # 为每个 source 过滤并导出 parquet
    for src in sources:
        safe = sanitize_name(src)
        out_dir = os.path.join(OUTPUT_DIR, safe)
        ensure_dir(out_dir)

        # 导出文件名：<split>.parquet 例如 train.parquet / test.parquet
        out_path = os.path.join(out_dir, f"{split_name}.parquet")

        # 过滤出当前 source 的数据并导出
        sub = ds.filter(lambda e: str(e["source"]) == src)
        # 使用 pandas 导出（更通用）
        df = sub.to_pandas()
        df.to_parquet(out_path, index=False)
        print(f"[{split_name}] source={src!r} -> {out_path}  (rows={len(df)})")

def main():
    ensure_dir(OUTPUT_DIR)
    split_and_save("train", PARQUET_PATH_TRAIN)
    split_and_save("test", PARQUET_PATH_TEST)
    print("✅ Done. 所有按 source 切分后的 parquet 已写入：", OUTPUT_DIR)

if __name__ == "__main__":
    main()