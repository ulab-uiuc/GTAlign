import json
from collections import defaultdict
import math
import matplotlib.pyplot as plt

# -------- Pareto 判定（最大化）---------
def is_dominated(a, b, keys=('user_reward', 'llm_reward')):
    return all(b[k] >= a[k] for k in keys) and any(b[k] > a[k] for k in keys)

def equal_vector(a, b, keys=('user_reward', 'llm_reward')):
    return all(a.get(k) == b.get(k) for k in keys)

# -------- 工具：按 input 建索引 --------
def normalize_input(x: str) -> str:
    return x.strip()

def build_map(records, key_field="input"):
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

# -------- 前沿与指标 --------
def pareto_frontier(points, keys=('user_reward', 'llm_reward')):
    non_dominated = []
    for i, a in enumerate(points):
        dominated = False
        for j, b in enumerate(points):
            if i == j:
                continue
            if is_dominated(a, b, keys):
                dominated = True
                break
        if not dominated:
            non_dominated.append(a)
    non_dominated.sort(key=lambda r: (r[keys[0]], r[keys[1]]))
    cleaned, cur_best = [], -math.inf
    for r in non_dominated:
        if r[keys[1]] >= cur_best:
            cleaned.append(r)
            cur_best = r[keys[1]]
    return cleaned

def coverage(setA, setB, keys=('user_reward','llm_reward')):
    if len(setB) == 0:
        return 0.0
    dominated_cnt = 0
    for b in setB:
        if any(is_dominated(b, a, keys) for a in setA):
            dominated_cnt += 1
    return dominated_cnt / len(setB)

def hypervolume(points, ref=(0.0, 0.0), keys=('user_reward','llm_reward')):
    if not points:
        return 0.0
    hv = 0.0
    xs = [ref[0]] + [p[keys[0]] for p in points]
    ys = [ref[1]] + [p[keys[1]] for p in points]
    for i in range(1, len(xs)):
        width = max(0.0, xs[i] - xs[i-1])
        height = max(0.0, ys[i])
        hv += width * height
    return hv

# -------- 主逻辑（总体对比，不画图）---------
def compare_corresponding(records1, records2,
                          key_field="input",
                          reward_keys=('user_reward', 'llm_reward')):
    map1, dup1 = build_map(records1, key_field=key_field)
    map2, dup2 = build_map(records2, key_field=key_field)

    common = set(map1.keys()) & set(map2.keys())
    if not common:
        raise ValueError("两个文件没有共有的测例（按 input 匹配后为空）。")

    f1_better = []
    f2_better = []
    equal_list = []
    incomparable = []

    for k in common:
        a = map1[k]
        b = map2[k]
        for rk in reward_keys:
            if rk not in a or rk not in b:
                raise KeyError(f"测例 '{k}' 缺少 reward 字段 {rk}")
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

    print(f"共有测例数：{len(common)}")
    print(f"文件1 支配 文件2 的数量：{len(f1_better)}")
    print(f"文件2 支配 文件1 的数量：{len(f2_better)}")
    print(f"两侧完全相等的数量：{len(equal_list)}")
    print(f"互不支配（不可比）的数量：{len(incomparable)}")

    return {
        "common_count": len(common),
        "file1_dominates_file2": sorted(f1_better),
        "file2_dominates_file1": sorted(f2_better),
        "equal": sorted(equal_list),
        "incomparable": sorted(incomparable),
        "duplicates_json1": {k: c for k, c in dup1.items() if c > 0},
        "duplicates_json2": {k: c for k, c in dup2.items() if c > 0}
    }

# -------- 新增：按 source 分组绘图 --------
def compare_and_plot_by_source(records1, records2,
                               key_field="input",
                               reward_keys=('user_reward','llm_reward'),
                               title1="Ours",
                               title2="Baseline",
                               save_dir=None,
                               show=True):
    """
    为每个 source 单独绘图；同图中画 Ours vs Baseline 的散点 + 全局前沿。
    若提供 save_dir，会把每个 source 的图保存为 PNG 文件（文件名含 source）。
    """
    map1, _ = build_map(records1, key_field=key_field)
    map2, _ = build_map(records2, key_field=key_field)
    common = set(map1.keys()) & set(map2.keys())
    if not common:
        raise ValueError("两个文件没有共有的测例（按 input 匹配后为空）。")

    # 仅保留共有测例，并按 source 分桶
    by_src_1 = defaultdict(list)
    by_src_2 = defaultdict(list)
    all_sources = set()

    for k in common:
        a = map1[k]; b = map2[k]
        src = a.get("source", b.get("source", "unknown"))
        all_sources.add(src)
        by_src_1[src].append(a)
        by_src_2[src].append(b)

    print(f"共有 source 数：{len(all_sources)} -> {sorted(all_sources)}")

    for src in sorted(all_sources):
        pts1 = by_src_1.get(src, [])
        pts2 = by_src_2.get(src, [])
        if not pts1 and not pts2:
            continue

        # 计算该 source 下的前沿与指标（使用各自前沿 + 全局前沿）
        front1 = pareto_frontier(pts1, keys=reward_keys)
        front2 = pareto_frontier(pts2, keys=reward_keys)
        global_front = pareto_frontier(pts1 + pts2, keys=reward_keys)

        # 参考点（向左下略放一点）
        all_u = [p[reward_keys[0]] for p in pts1 + pts2]
        all_l = [p[reward_keys[1]] for p in pts1 + pts2]
        ru = min(all_u) - 0.01 * (max(all_u) - min(all_u) + 1e-12)
        rl = min(all_l) - 0.01 * (max(all_l) - min(all_l) + 1e-12)
        ref = (ru, rl)

        hv1 = hypervolume(front1, ref=ref, keys=reward_keys)
        hv2 = hypervolume(front2, ref=ref, keys=reward_keys)
        cov12 = coverage(pts1, pts2, keys=reward_keys)
        cov21 = coverage(pts2, pts1, keys=reward_keys)

        # ---- 绘图（单图）----
        fig, ax = plt.subplots(figsize=(7, 6))

        # 散点：Ours vs Baseline（不同 marker；不强制颜色，让 Matplotlib 自动配色）
        if pts1:
            ax.scatter([p[reward_keys[0]] for p in pts1],
                       [p[reward_keys[1]] for p in pts1],
                       label=f"{title1} (n={len(pts1)})", marker='o', alpha=0.6)
        if pts2:
            ax.scatter([p[reward_keys[0]] for p in pts2],
                       [p[reward_keys[1]] for p in pts2],
                       label=f"{title2} (n={len(pts2)})", marker='^', alpha=0.6)

        # 前沿折线
        def plot_front(front, label, linestyle):
            if not front: return
            xs = [p[reward_keys[0]] for p in front]
            ys = [p[reward_keys[1]] for p in front]
            ax.plot(xs, ys, linestyle=linestyle, linewidth=2.0, label=label)

        plot_front(front1, f"{title1} frontier", '-')
        plot_front(front2, f"{title2} frontier", '--')
        plot_front(global_front, "Global frontier", ':')

        ax.scatter([ref[0]], [ref[1]], marker='x', s=60, label='Ref point')
        ax.set_xlabel(reward_keys[0])
        ax.set_ylabel(reward_keys[1])
        ax.grid(True, linestyle=':')

        # 标题中直接展示关键指标
        ax.set_title(
            f"Source: {src} | HV({title1})={hv1:.4f}, HV({title2})={hv2:.4f}, "
            f"ΔHV={hv1-hv2:.4f} | C({title1},{title2})={cov12:.3f}, C({title2},{title1})={cov21:.3f}"
        )
        ax.legend(loc='best')
        fig.tight_layout()

        if save_dir:
            safe_src = str(src).replace('/', '_').replace(' ', '_')
            path = f"{save_dir}/pareto_scatter_{safe_src}.png"
            plt.savefig(path, dpi=200)
            print(f"[Saved] {path}")
        if show:
            plt.show()
        else:
            plt.close(fig)

import csv

def compare_by_source(records1, records2,
                      key_field="input",
                      reward_keys=('user_reward','llm_reward'),
                      source_field="source",
                      csv_path=None):
    """
    返回一个 dict: { source: { 'total', 'ours_dominates', 'baseline_dominates', 'equal', 'incomparable' } }
    若提供 csv_path，则将结果写入 CSV。
    """
    map1, _ = build_map(records1, key_field=key_field)
    map2, _ = build_map(records2, key_field=key_field)
    common = set(map1.keys()) & set(map2.keys())
    if not common:
        raise ValueError("两个文件没有共有的测例（按 input 匹配后为空）。")

    # 按 source 分桶（只保留共有测例）
    buckets = defaultdict(list)  # source -> list of input keys
    for k in common:
        a = map1[k]; b = map2[k]
        src = a.get(source_field, b.get(source_field, "unknown"))
        buckets[src].append(k)

    results = {}
    for src, keys_in_src in buckets.items():
        ours_dom, base_dom, equal_cnt, incomparable_cnt = 0, 0, 0, 0
        for k in keys_in_src:
            a = map1[k]; b = map2[k]
            # 字段检查
            for rk in reward_keys:
                if rk not in a or rk not in b:
                    raise KeyError(f"测例 '{k}' 缺少 reward 字段 {rk}")

            adom_b = is_dominated(b, a, reward_keys)  # A 支配 B?（注意函数签名）
            bdom_a = is_dominated(a, b, reward_keys)  # B 支配 A?

            if adom_b and bdom_a:
                # 理论不可能，这里记为不可比并继续
                incomparable_cnt += 1
            elif adom_b:
                ours_dom += 1
            elif bdom_a:
                base_dom += 1
            elif equal_vector(a, b, reward_keys):
                equal_cnt += 1
            else:
                incomparable_cnt += 1

        results[src] = {
            "total": len(keys_in_src),
            "ours_dominates": ours_dom,
            "baseline_dominates": base_dom,
            "equal": equal_cnt,
            "incomparable": incomparable_cnt
        }

    # 可选导出 CSV
    if csv_path:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["source", "total", "ours_dominates", "baseline_dominates", "equal", "incomparable"])
            for src in sorted(results.keys()):
                r = results[src]
                writer.writerow([src, r["total"], r["ours_dominates"], r["baseline_dominates"], r["equal"], r["incomparable"]])
        print(f"[Saved] per-source dominance stats -> {csv_path}")

    # 控制台友好打印
    print("Per-source Pareto dominance (Ours vs Baseline):")
    for src in sorted(results.keys()):
        r = results[src]
        print(f"- {src}: total={r['total']}, Ours>{r['ours_dominates']}, Baseline>{r['baseline_dominates']}, equal={r['equal']}, incomparable={r['incomparable']}")

    return results

# -------- 示例用法 --------
if __name__ == "__main__":
    json1_path = '/mnt/bn/heheda/verl/ppo_0909_bs1024/21.jsonl'  # 我们的方法
    json2_path = '/mnt/bn/heheda/verl/ppo_0909_bs1024_linear_comb/21.jsonl'                      # 基线

    json1 = []
    with open(json1_path, 'r') as f:
        for line in f:
            json1.append(json.loads(line))

    json2 = []
    with open(json2_path, 'r') as f:
        for line in f:
            json2.append(json.loads(line))

    # 仍可先跑一次总体对比
    # _ = compare_corresponding(json1, json2)

    # 按 source 分组绘图（可指定保存目录）
    # compare_and_plot_by_source(
    #     json1, json2,
    #     key_field="input",
    #     reward_keys=('user_reward', 'llm_reward'),
    #     title1="Ours",
    #     title2="Baseline",
    #     save_dir="/mnt/bn/heheda/verl/recipe/gamellm/experiments",   # 例如 "plots/source_figs"
    #     show=True
    # )
    stats = compare_by_source(
        json1, json2,
        key_field="input",
        reward_keys=('user_reward','llm_reward'),
        source_field="source",
        csv_path="/mnt/bn/heheda/verl/recipe/gamellm/experiments/per_source_pareto_stats.csv"  # 可选
    )