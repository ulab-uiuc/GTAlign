from datasets import load_dataset, DatasetDict, Features, Sequence, Value

SEED = 42

sampled = load_dataset("zsqzz/wildguard-adversarial-harmful-3k", split="train")

def transform_with_id(example, idx):
    p = example.get("prompt", "") or ""
    phl = example.get("prompt_harm_label", "") or ""
    return {
        "prompt": [{"role": "user", "content": p}],
        "source": "wildguard",
        "data_source": "wildguard",
        "ability": "wildguard",
        "single_turn_prompt": p,
        "single_turn_completion": phl,
        "conv_id": idx,          # 如需从1开始：idx + 1
        "unique_conv_id": idx,   # 如需从1开始：idx + 1
    }

new_train = sampled.map(
    transform_with_id,
    with_indices=True,
    remove_columns=sampled.column_names
)

new_ds = DatasetDict({"train": new_train})
# new_train.to_json("wildguard_adversarial_harmful_3k.jsonl", lines=True, orient="records")
# new_train.to_parquet("wildguard_adversarial_harmful_3k.parquet")
new_ds.push_to_hub("zsqzz/wildguard-3k-formal")
print("Done. Samples:", len(new_train))