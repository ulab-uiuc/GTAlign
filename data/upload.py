# upload_parquet_to_huggingface.py

from datasets import load_dataset, DatasetDict
from huggingface_hub import login

# ====== 修改以下配置 ======
HF_TOKEN = "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"  # 你的 Hugging Face Token（建议保密处理）
REPO_NAME = "zsqzz/gamellm_train_medium"  # 上传后的数据集名称
PARQUET_PATH = "/mnt/bn/heheda/verl/data/gamellm_medium/train.parquet"  # 本地 .parquet 文件路径
# =========================

def main():
    # 登录 Hugging Face（只需要运行一次）
    login(token=HF_TOKEN)

    # 加载本地 .parquet 文件为 HuggingFace Dataset
    dataset = load_dataset("parquet", data_files=PARQUET_PATH, split="train")

    # 可选：构造 DatasetDict（如果你有多个 split）
    dataset_dict = DatasetDict({"train": dataset})

    # 上传到 Hugging Face Hub
    print(f"Pushing dataset to: {REPO_NAME}")
    dataset_dict.push_to_hub(REPO_NAME)

    print("✅ Upload successful.")

if __name__ == "__main__":
    main()