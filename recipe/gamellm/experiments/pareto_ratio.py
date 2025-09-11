import json
from collections import defaultdict

# -------- Pareto 判定（最大化）---------
def is_dominated(a, b, keys=('user_reward', 'llm_reward')):
    """
    判断 a 是否被 b 支配：b 在 keys 各维度都 >= a，且至少一维 >
    """
    return all(b[k] >= a[k] for k in keys) and any(b[k] > a[k] for k in keys)

def equal_vector(a, b, keys=('user_reward', 'llm_reward')):
    return all(a.get(k) == b.get(k) for k in keys)

# -------- 工具：按 input 建索引 --------
def normalize_input(x: str) -> str:
    return x.strip()

def build_map(records, key_field="input"):
    """
    将列表[dict,...] -> 映射: norm(input) -> 该条记录
    如果同一个 input 出现多次，默认保留最后一条；同时统计重复。
    """
    mp = {}
    dup_counter = defaultdict(int)
    for r in records:
        if key_field not in r:
            raise KeyError(f"记录缺少 '{key_field}' 字段: {r}")
        k = normalize_input(r[key_field])
        if k in mp:
            dup_counter[k] += 1
        mp[k] = r
    return mp, dup_counter

# -------- 主逻辑：两文件“对应项”逐一比较 --------
def compare_corresponding(records1, records2,
                          key_field="input",
                          reward_keys=('user_reward', 'llm_reward'),
                          score_key="score"):
    map1, dup1 = build_map(records1, key_field=key_field)
    map2, dup2 = build_map(records2, key_field=key_field)

    common = set(map1.keys()) & set(map2.keys())
    if not common:
        raise ValueError("两个文件没有共有的测例（按 input 匹配后为空）。")

    f1_better = []
    f2_better = []
    equal_list = []
    incomparable = []

    # -------- 新增：score 统计 --------
    score_stats = {
        "file1": {">0": 0, "=0": 0, "<0": 0},
        "file2": {">0": 0, "=0": 0, "<0": 0}
    }

    for k in common:
        a = map1[k]
        b = map2[k]
        for rk in reward_keys:
            if rk not in a or rk not in b:
                raise KeyError(f"测例 '{k}' 缺少 reward 字段 {rk}")

        # ====== Pareto 判定 ======
        if is_dominated(a, b, reward_keys) and is_dominated(b, a, reward_keys):
            incomparable.append(k)
            raise ValueError(f"测例 '{k}' 同时被文件1和文件2支配，这是理论上不可能的。")
        elif is_dominated(b, a, reward_keys):
            f1_better.append(k)
        elif is_dominated(a, b, reward_keys):
            f2_better.append(k)
        elif equal_vector(a, b, reward_keys):
            equal_list.append(k)
        else:
            incomparable.append(k)

        # ====== 统计 score ======
        if score_key in a:
            if a[score_key] > b[score_key]:
                score_stats["file1"][">0"] += 1
            elif a[score_key] == b[score_key]:
                score_stats["file1"]["=0"] += 1
            else:
                score_stats["file1"]["<0"] += 1

        # if score_key in b:
        #     if b[score_key] > 0:
        #         score_stats["file2"][">0"] += 1
        #     elif b[score_key] == 0:
        #         score_stats["file2"]["=0"] += 1
        #     else:
        #         score_stats["file2"]["<0"] += 1

    print(f"共有测例数：{len(common)}")
    print(f"文件1 支配 文件2 的数量：{len(f1_better)}")
    print(f"文件2 支配 文件1 的数量：{len(f2_better)}")
    print(f"两侧完全相等的数量：{len(equal_list)}")
    print(f"互不支配（不可比）的数量：{len(incomparable)}")
    print("文件1 score 统计：", score_stats["file1"])
    print("文件2 score 统计：", score_stats["file2"])

    return {
        "common_count": len(common),
        "file1_dominates_file2": sorted(f1_better),
        "file2_dominates_file1": sorted(f2_better),
        "equal": sorted(equal_list),
        "incomparable": sorted(incomparable),
        "duplicates_json1": {k: c for k, c in dup1.items() if c > 0},
        "duplicates_json2": {k: c for k, c in dup2.items() if c > 0},
        "score_stats": score_stats
    }

# -------- 用法示例（沿用你的读文件方式）---------
if __name__ == "__main__":
    json1 = []
    with open('/mnt/bn/heheda/verl/ppo_0909_bs1024_llmreward/22.jsonl', 'r') as f:
        for line in f:
            tmp = json.loads(line)
            # if tmp['source'] != "ambig_qa":
            json1.append(tmp)

    json2 = []
    with open('/mnt/bn/heheda/verl/ppo_0909_bs1024_linear_comb/22.jsonl', 'r') as f:
        for line in f:
            tmp = json.loads(line)
            # if tmp['source'] != "ambig_qa":
            json2.append(tmp)

    result = compare_corresponding(json1, json2)
    # print(result)