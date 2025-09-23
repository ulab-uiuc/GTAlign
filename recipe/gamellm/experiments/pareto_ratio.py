#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pareto_compare_plus.py

Compare two result sets on (user_reward, llm_reward) with richer Pareto-style metrics:
- Strict dominance counts (existing logic)
- ε-Pareto coverage vs. joint frontier
- Hypervolume (2D) against a data-driven reference point
- Average regret to the joint frontier (normalized, L_inf)
- Social welfare aggregates: Nash Social Welfare (geometric mean) and Cobb–Douglas
- Coverage / spread (range coverage across both axes)

Input format: two JSONL files, each line is a dict including at least:
  { "input": <str>, "user_reward": <float>, "llm_reward": <float>, ... }

Only shared keys by 'input' across the two files are compared/evaluated.

Usage (edit file paths in __main__ or run as module and call main(...)).
"""

import json
import math
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

# -------- Pareto 判定（最大化）---------
# def is_dominated(a: Dict, b: Dict, keys: Sequence[str] = ('user_reward', 'llm_reward')) -> bool:
#     """
#     判断 a 是否被 b 支配：b 在 keys 各维度都 >= a，且至少一维 >
#     """
#     return all(b[k] >= a[k] for k in keys) and any(b[k] >= a[k] for k in keys)

from typing import Dict, Sequence

def is_dominated(
    a: Dict[str, float],
    b: Dict[str, float],
    keys: Sequence[str] = ('user_reward', 'llm_reward'),
    eps: float = 0.05
) -> bool:
    """
    判断 a 是否被 b ε-支配：
    b 在各维度都 >= a - eps，且至少一维 > a - eps
    """
    return all(b[k] >= a[k] - eps for k in keys) and any(b[k] > a[k] - eps for k in keys)

def equal_vector(a: Dict, b: Dict, keys: Sequence[str] = ('user_reward', 'llm_reward')) -> bool:
    return all(a.get(k) == b.get(k) for k in keys)

# -------- 工具：按 input 建索引 --------
def normalize_input(x: str) -> str:
    return x.strip()

def build_map(records: List[Dict], key_field: str = "input") -> Tuple[Dict[str, Dict], Dict[str, int]]:
    """
    将列表[dict,...] -> 映射: norm(input) -> 该条记录
    如果同一个 input 出现多次，默认保留最后一条；同时统计重复。
    """
    mp: Dict[str, Dict] = {}
    dup_counter: Dict[str, int] = defaultdict(int)
    for r in records:
        if key_field not in r:
            raise KeyError(f"记录缺少 '{key_field}' 字段: {r}")
        k = normalize_input(str(r[key_field]))
        if k in mp:
            dup_counter[k] += 1
        mp[k] = r
    return mp, dup_counter

# -------- 前沿与指标工具 --------
def pareto_frontier(points: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    给定二维点集（最大化），返回非支配（Pareto）前沿。
    实现：按 user_reward 降序排序，逐一扫描维护 llm_reward 的最大值。
    """
    pts = sorted(points, key=lambda x: (-x[0], -x[1]))
    frontier: List[Tuple[float, float]] = []
    best_llm = -float('inf')
    for u, l in pts:
        if l > best_llm:
            frontier.append((u, l))
            best_llm = l
    return frontier

def hypervolume_2d(frontier: List[Tuple[float, float]], reference: Tuple[float, float]) -> float:
    """
    2D 超体积（最大化）：以 reference=(r_u, r_l) 为参考点，计算前沿与参考点围成的面积。
    要求 frontier 已按 user_reward 降序且是非支配集（可用 pareto_frontier 的输出）。
    """
    if not frontier:
        return 0.0
    r_u, r_l = reference
    hv = 0.0
    prev_u = r_u
    for (u, l) in frontier:
        width = max(0.0, u - prev_u)
        height = max(0.0, l - r_l)
        hv += width * height
        prev_u = u
    # 加上最后一段从最小的 u 到参考 u 的面积（按排序我们从大到小走，已累积完）
    return hv

def rectified(value: float) -> float:
    return value if value > 0 else 0.0

def l_inf_regret_to_frontier(point: Tuple[float, float],
                             frontier: List[Tuple[float, float]],
                             ranges: Tuple[float, float]) -> float:
    """
    与联合前沿的 L_inf 归一化 regret：
      regret = min_{p* in frontier} max( (u*-u)/range_u, (l*-l)/range_l, 0 )
    """
    u, l = point
    range_u, range_l = ranges
    best = float('inf')
    for fu, fl in frontier:
        du = rectified(fu - u) / (range_u if range_u > 0 else 1.0)
        dl = rectified(fl - l) / (range_l if range_l > 0 else 1.0)
        best = min(best, max(du, dl))
    # 如果点本身是前沿上的，regret 应为 0
    return best if best != float('inf') else 0.0

def within_epsilon_of_frontier(point: Tuple[float, float],
                               frontier: List[Tuple[float, float]],
                               ranges: Tuple[float, float],
                               eps: float = 0.05) -> bool:
    """
    近似帕累托(ε)：存在前沿点 p* 使得
      u >= u* - eps*range_u 且 l >= l* - eps*range_l
    等价：max( (u*-u)/range_u, (l*-l)/range_l ) <= eps  且两项都非负
    """
    u, l = point
    range_u, range_l = ranges
    if not frontier:
        return False
    for fu, fl in frontier:
        du = (fu - u) / (range_u if range_u > 0 else 1.0)
        dl = (fl - l) / (range_l if range_l > 0 else 1.0)
        if du <= eps and dl <= eps:
            return True
    return False

def aggregate_social_welfare(points: Iterable[Tuple[float, float]],
                             alpha: float = 0.5,
                             epsilon: float = 1e-9) -> Tuple[float, float]:
    """
    返回 (平均 Nash Social Welfare, 平均 Cobb-Douglas^alpha )：
      NSW = sqrt( (u+eps)*(l+eps) )
      CD  = (u+eps)^alpha * (l+eps)^(1-alpha)
    这里用几何平均的形式（NSW 用 sqrt 只是等价缩放，不影响比较）。
    """
    pts = list(points)
    if not pts:
        return 0.0, 0.0
    nsw_sum = 0.0
    cd_sum = 0.0
    for u, l in pts:
        nsw_sum += math.sqrt(max(u, 0.0) + epsilon) * math.sqrt(max(l, 0.0) + epsilon)
        cd_sum += (max(u, 0.0) + epsilon) ** alpha * (max(l, 0.0) + epsilon) ** (1.0 - alpha)
    m = float(len(pts))
    return nsw_sum / m, cd_sum / m

def coverage_spread(points: Iterable[Tuple[float, float]],
                    global_min: Tuple[float, float],
                    global_max: Tuple[float, float]) -> Tuple[float, float, float]:
    """
    返回 (u覆盖率, l覆盖率, 平均覆盖率)。覆盖率 = (max - min) / 全局范围。
    """
    pts = list(points)
    if not pts:
        return 0.0, 0.0, 0.0
    us = [p[0] for p in pts]
    ls = [p[1] for p in pts]
    u_min, l_min = global_min
    u_max, l_max = global_max
    u_span = (max(us) - min(us)) if len(us) > 1 else 0.0
    l_span = (max(ls) - min(ls)) if len(ls) > 1 else 0.0
    u_cov = u_span / (u_max - u_min) if u_max > u_min else 0.0
    l_cov = l_span / (l_max - l_min) if l_max > l_min else 0.0
    return u_cov, l_cov, 0.5 * (u_cov + l_cov)

# -------- 主逻辑：两文件“对应项”逐一比较 --------
def compare_corresponding(records1: List[Dict], records2: List[Dict],
                          key_field: str = "input",
                          reward_keys: Sequence[str] = ('user_reward', 'llm_reward'), eps: float = 0.05) -> Dict:
    map1, dup1 = build_map(records1, key_field=key_field)
    map2, dup2 = build_map(records2, key_field=key_field)

    common = set(map1.keys()) & set(map2.keys())
    if not common:
        raise ValueError("两个文件没有共有的测例（按 input 匹配后为空）。")

    f1_better: List[str] = []
    f2_better: List[str] = []
    equal_list: List[str] = []
    incomparable: List[str] = []

    for k in common:
        a = map1[k]
        b = map2[k]
        for rk in reward_keys:
            if rk not in a or rk not in b:
                raise KeyError(f"测例 '{k}' 缺少 reward 字段 {rk}")

        # ====== Pareto 判定 ======
        if is_dominated(a, b, reward_keys, eps) and is_dominated(b, a, reward_keys, eps):
            incomparable.append(k)
            # raise ValueError(f"测例 '{k}' 同时被文件1和文件2支配，这是理论上不可能的。")
        elif is_dominated(b, a, reward_keys, eps):
            f1_better.append(k)
        elif is_dominated(a, b, reward_keys, eps):
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
        "duplicates_json2": {k: c for k, c in dup2.items() if c > 0},
        "map1": map1,
        "map2": map2,
        "common_keys": sorted(common),
    }

# -------- 高阶指标计算 --------
def compute_metrics(records1: List[Dict], records2: List[Dict],
                    key_field: str = "input",
                    reward_keys: Sequence[str] = ('user_reward', 'llm_reward'),
                    eps: float = 0.05,
                    alpha: float = 0.5) -> Dict:
    cmp_res = compare_corresponding(records1, records2, key_field=key_field, reward_keys=reward_keys, eps=eps)
    map1 = cmp_res["map1"]
    map2 = cmp_res["map2"]
    keys = cmp_res["common_keys"]

    # Build point sets (common only)
    pts1 = [(float(map1[k][reward_keys[0]]), float(map1[k][reward_keys[1]])) for k in keys]
    pts2 = [(float(map2[k][reward_keys[0]]), float(map2[k][reward_keys[1]])) for k in keys]
    all_pts = pts1 + pts2

    # Global ranges, min/max for normalization
    u_vals = [p[0] for p in all_pts]
    l_vals = [p[1] for p in all_pts]
    u_min, u_max = min(u_vals), max(u_vals)
    l_min, l_max = min(l_vals), max(l_vals)
    range_u = u_max - u_min
    range_l = l_max - l_min

    # Joint frontier (based on union of both methods)
    joint_front = pareto_frontier(all_pts)

    # ε-Pareto coverage: % of points within eps of joint frontier
    eps_cov_1 = sum(within_epsilon_of_frontier(p, joint_front, (range_u, range_l), eps=eps) for p in pts1) / max(1, len(pts1))
    eps_cov_2 = sum(within_epsilon_of_frontier(p, joint_front, (range_u, range_l), eps=eps) for p in pts2) / max(1, len(pts2))

    # Regret (L_inf to joint frontier), averaged
    regrets_1 = [l_inf_regret_to_frontier(p, joint_front, (range_u, range_l)) for p in pts1]
    regrets_2 = [l_inf_regret_to_frontier(p, joint_front, (range_u, range_l)) for p in pts2]
    avg_regret_1 = sum(regrets_1) / max(1, len(regrets_1))
    avg_regret_2 = sum(regrets_2) / max(1, len(regrets_2))

    # Hypervolume (2D) vs reference. Choose reference as (u_min, l_min)
    front1 = pareto_frontier(pts1)
    front2 = pareto_frontier(pts2)
    ref = (u_min, l_min)
    hv1 = hypervolume_2d(front1, ref)
    hv2 = hypervolume_2d(front2, ref)

    # Social welfare aggregates
    nsw1, cd1 = aggregate_social_welfare(pts1, alpha=alpha)
    nsw2, cd2 = aggregate_social_welfare(pts2, alpha=alpha)

    # Coverage / spread
    u_cov1, l_cov1, cov1 = coverage_spread(pts1, (u_min, l_min), (u_max, l_max))
    u_cov2, l_cov2, cov2 = coverage_spread(pts2, (u_min, l_min), (u_max, l_max))

    return {
        "basic_compare": {
            "common_count": cmp_res["common_count"],
            "file1_dominates_file2": len(cmp_res["file1_dominates_file2"]),
            "file2_dominates_file1": len(cmp_res["file2_dominates_file1"]),
            "equal": len(cmp_res["equal"]),
            "incomparable": len(cmp_res["incomparable"]),
        },
        "ranges": {"u_min": u_min, "u_max": u_max, "l_min": l_min, "l_max": l_max},
        "epsilon": eps,
        "epsilon_coverage": {"file1": eps_cov_1, "file2": eps_cov_2},
        "avg_regret_Linf": {"file1": avg_regret_1, "file2": avg_regret_2},
        "hypervolume_ref_(u_min,l_min)": {"file1": hv1, "file2": hv2},
        "social_welfare_alpha": alpha,
        "NSW_avg": {"file1": nsw1, "file2": nsw2},
        "CobbDouglas_avg": {"file1": cd1, "file2": cd2},
        "coverage_spread": {
            "file1": {"u_cov": u_cov1, "l_cov": l_cov1, "mean_cov": cov1},
            "file2": {"u_cov": u_cov2, "l_cov": l_cov2, "mean_cov": cov2},
        },
    }

def pretty_print_metrics(m: Dict) -> None:
    print("\n==== 基础严格帕累托统计（逐输入比对） ====")
    b = m["basic_compare"]
    print(f"共有测例数: {b['common_count']}")
    print(f"文件1 支配 文件2: {b['file1_dominates_file2']}")
    print(f"文件2 支配 文件1: {b['file2_dominates_file1']}")
    print(f"两侧完全相等: {b['equal']}")
    print(f"互不支配（不可比）: {b['incomparable']}")

    print("\n==== ε-Pareto 覆盖率（相对联合前沿, eps={:.3f}） ====".format(m["epsilon"]))
    print(f"文件1: {m['epsilon_coverage']['file1']:.3f}")
    print(f"文件2: {m['epsilon_coverage']['file2']:.3f}")

    print("\n==== 平均 Regret 到联合前沿（L_inf, 归一化） ====")
    print(f"文件1: {m['avg_regret_Linf']['file1']:.4f}")
    print(f"文件2: {m['avg_regret_Linf']['file2']:.4f}")

    ref = m["ranges"]
    print("\n==== 超体积 Hypervolume（参考点 = (u_min, l_min) = ({:.4f}, {:.4f})） ====".format(ref["u_min"], ref["l_min"]))
    print(f"文件1: {m['hypervolume_ref_(u_min,l_min)']['file1']:.6f}")
    print(f"文件2: {m['hypervolume_ref_(u_min,l_min)']['file2']:.6f}")

    print("\n==== 社会福利（平均值） ====")
    print(f"NSW 平均: 文件1={m['NSW_avg']['file1']:.6f} | 文件2={m['NSW_avg']['file2']:.6f}")
    print(f"Cobb–Douglas(α={m['social_welfare_alpha']}) 平均: 文件1={m['CobbDouglas_avg']['file1']:.6f} | 文件2={m['CobbDouglas_avg']['file2']:.6f}")

    print("\n==== 覆盖/分布（相对全局范围的覆盖率） ====")
    cov1 = m["coverage_spread"]["file1"]
    cov2 = m["coverage_spread"]["file2"]
    print("文件1: u_cov={:.3f}, l_cov={:.3f}, mean_cov={:.3f}".format(cov1["u_cov"], cov1["l_cov"], cov1["mean_cov"]))
    print("文件2: u_cov={:.3f}, l_cov={:.3f}, mean_cov={:.3f}".format(cov2["u_cov"], cov2["l_cov"], cov2["mean_cov"]))


# -------- 用法示例（沿用你的读文件方式，可直接修改路径）---------
def load_jsonl(path: str) -> List[Dict]:
    out: List[Dict] = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out

if __name__ == "__main__":
    # ==== 修改为你的文件路径 ====
    path1 = '/mnt/bn/heheda/verl/0910_wildguard_cobb/140.jsonl'  # 方法1
    path2 = '/mnt/bn/heheda/verl/0910_wildguard_llm/140.jsonl'    # 方法2

    json1 = load_jsonl(path1)
    json2 = load_jsonl(path2)

    # 计算各类指标
    metrics = compute_metrics(json1, json2, key_field="input", reward_keys=('user_reward', 'llm_reward'),
                              eps=0,  # 5% 的 ε-近似帕累托
                              alpha=0.5) # Cobb–Douglas 中的 α，可按需调整

    # 漂亮地打印结果
    pretty_print_metrics(metrics)
