# upload_parquet_to_huggingface.py

from datasets import load_dataset, DatasetDict
from huggingface_hub import login

# ====== 修改以下配置 ======
HF_TOKEN = "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"  # 建议通过环境变量读取，而不是写死
REPO_NAME = "zsqzz/gamellm_wildbreak_math_medium_abgqa"
PARQUET_PATH_TEST = "/mnt/bn/heheda/verl/data/gamellm_wildbreak_math_medium_abgqa/test.parquet"
PARQUET_PATH_TRAIN = "/mnt/bn/heheda/verl/data/gamellm_wildbreak_math_medium_abgqa/train.parquet"
# =========================

def main():
    # 登录 Hugging Face
    login(token=HF_TOKEN)

    # 加载本地 parquet 文件
    dataset_train = load_dataset("parquet", data_files=PARQUET_PATH_TRAIN, split="train")
    dataset_test = load_dataset("parquet", data_files=PARQUET_PATH_TEST, split="train")

    # 组织成 DatasetDict（train/test 两个 split）
    dataset_dict = DatasetDict({
        "train": dataset_train,
        "test": dataset_test,
    })

    # 上传到 Hugging Face Hub
    print(f"Pushing dataset to: {REPO_NAME}")
    dataset_dict.push_to_hub(REPO_NAME)

    print("✅ Upload successful.")

if __name__ == "__main__":
    main()