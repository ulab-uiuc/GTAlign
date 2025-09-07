from datasets import load_dataset, DatasetDict

SEED = 42

# 载入原始数据
sampled = load_dataset("sewon/ambig_qa", split="train")

def transform_with_id(example, idx):
    p = example.get("question", "") or ""
    resp = example["annotations"]["type"][0]
    if resp == "singleAnswer":
        resp += '###'
        resp += example["annotations"]["answer"][0][0]
    elif resp == "multipleQAs":
        resp += '###There are multiple meanings of the question.'
        for idx, item in enumerate(example["annotations"]["qaPairs"][0]["question"]):
            resp += "Meaning " + str(idx) + ": " + item
            resp += "; Answer " + str(idx) + ": " + example["annotations"]["qaPairs"][0]["answer"][idx][0] + '\n'
    return {
        "prompt": [{"role": "user", "content": p}],
        "source": "ambig_qa",
        "data_source": "ambig_qa",
        "ability": "ambig_qa",
        "single_turn_prompt": p,
        "single_turn_completion": resp,
        "conv_id": idx,
        "unique_conv_id": idx,
    }

# 先采样 3000 条，再做 transform
sampled_3k = sampled.shuffle(seed=SEED).select(range(3000))

new_train = sampled_3k.map(
    transform_with_id,
    with_indices=True,
    remove_columns=sampled.column_names
)

new_ds = DatasetDict({"train": new_train})

# push 到 hub
new_ds.push_to_hub("zsqzz/abgqa-3k-formal")
print("Done. Samples:", len(new_train))