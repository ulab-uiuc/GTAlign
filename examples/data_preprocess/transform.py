from datasets import load_dataset
from huggingface_hub import login

# 如果需要认证
# login("your_hf_token")

# 1) 读取 dataset
dataset = load_dataset("zsqzz/rl-mathhard-medium-merged-0902", split="train")

# 2) 修改某个 key 的值
# 举例：把 "source" 字段全改成 "modified"
def modify_key(example):
    new_p = []
    for t in example["prompt"]:
        if t["role"] == "user":
            new_p.append({"role": "user", "content": t["content"]})
        else:
            new_p.append({"role": "assistant", "content": t["content"]})
    example["prompt"] = new_p
    return example

prompt_dict = {}
for item in dataset:
    example = modify_key(item)
    prompt_dict[item["unique_conv_id"]] = example["prompt"]

dataset = dataset.remove_columns(["prompt"])
dataset = dataset.map(lambda x: {"prompt": prompt_dict[x["unique_conv_id"]]})
# 3) push 回 HuggingFace Hub
# 方法 1：直接用 push_to_hub（推荐）
dataset.push_to_hub("zsqzz/rl-mathhard-medium-merged-0902-new")

# 方法 2：保存成本地，再手动上传
# dataset.save_to_disk("tmp_dataset")