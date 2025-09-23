import unicodedata
from typing import Dict, List, Tuple

def _prep(
    s: str,
    ignore_whitespace: bool = True,
    ignore_case: bool = False,
    normalize_unicode: bool = True,
) -> str:
    if s is None:
        return ""
    if normalize_unicode:
        s = unicodedata.normalize("NFKC", s)
    if ignore_whitespace:
        s = "".join(ch for ch in s if not ch.isspace())
    if ignore_case:
        s = s.lower()
    return s

def ngram_repeat_runs(
    s: str,
    n_min: int = 2,
    n_max: int = 6,
    *,
    ignore_whitespace: bool = True,
    ignore_case: bool = False,
    normalize_unicode: bool = True,
) -> List[Tuple[int, int, int]]:
    """
    返回所有 n-gram 连续重复段的列表: [(start_idx, n, k), ...]
      - n: n-gram 的长度
      - k: 重复的次数（>=2）
    规则：仅统计“相邻且完全相等”的重复块（back-to-back），并在同一 n 上进行跳跃以避免自重叠。
    """
    s2 = _prep(s, ignore_whitespace, ignore_case, normalize_unicode)
    L = len(s2)
    if L == 0:
        return []
    runs = []
    for n in range(max(1, n_min), max(1, n_max) + 1):
        if n * 2 > L:
            continue
        i = 0
        while i + 2 * n <= L:
            # 以 s2[i:i+n] 为模板，向右数连续相等的块
            k = 1
            while i + (k + 1) * n <= L and s2[i + (k - 1) * n : i + k * n] == s2[i + k * n : i + (k + 1) * n]:
                k += 1
            if k >= 2:
                runs.append((i, n, k))
                i += n * k  # 跳过该段，防止同一 n 的自重叠
            else:
                i += 1
    return runs

def ngram_repeat_metrics(
    s: str,
    n_min: int = 2,
    n_max: int = 6,
    *,
    ignore_whitespace: bool = True,
    ignore_case: bool = False,
    normalize_unicode: bool = True,
) -> Dict[str, float]:
    """
    指标:
      - max_blocks: 单段重复的最大“块次数” k（>=2）
      - max_blocks_n: 对应的 n
      - max_span_len: 覆盖字符数 = n * k
      - coverage_ratio: 所有重复段覆盖的字符占整体比例（用区间并集避免重叠重复计数）
      - num_runs: 重复段数量
    """
    s2 = _prep(s, ignore_whitespace, ignore_case, normalize_unicode)
    L = len(s2)
    if L == 0:
        return dict(max_blocks=0.0, max_blocks_n=0.0, max_span_len=0.0,
                    coverage_ratio=0.0, num_runs=0.0, length=0.0)

    runs = ngram_repeat_runs(
        s2, n_min, n_max,
        ignore_whitespace=False,  # 已预处理
        ignore_case=False,
        normalize_unicode=False,
    )

    if not runs:
        return dict(max_blocks=1.0, max_blocks_n=0.0, max_span_len=1.0,
                    coverage_ratio=0.0, num_runs=0.0, length=float(L))

    # 最长块重复
    max_blocks, max_blocks_n = 0, 0
    for _, n, k in runs:
        if k > max_blocks or (k == max_blocks and n > max_blocks_n):
            max_blocks, max_blocks_n = k, n

    # 计算覆盖比例（区间并集）
    intervals = []
    for i, n, k in runs:
        intervals.append((i, i + n * k))
    intervals.sort()
    merged = []
    for s0, e0 in intervals:
        if not merged or s0 > merged[-1][1]:
            merged.append([s0, e0])
        else:
            merged[-1][1] = max(merged[-1][1], e0)
    covered = sum(e - s for s, e in merged)

    return dict(
        max_blocks=float(max_blocks),
        max_blocks_n=float(max_blocks_n),
        max_span_len=float(max_blocks * max_blocks_n if max_blocks_n else 0),
        coverage_ratio=float(covered / L),
        num_runs=float(len(runs)),
        length=float(L),
    )

def ngram_repeat_reward(
    s: str,
    n_min: int = 2,
    n_max: int = 6,
    *,
    allowed_max_blocks: int = 2,   # 允许同一 n 连起来最多出现几块（如 2 表示 "abcabc" 可接受）
    beta: float = 2.5,             # 对“最长块次数”超阈的敏感度
    alpha: float = 6.0,            # 对覆盖比例的敏感度
    w_blocks: float = 0.6,         # 两路加权，和为 1
    w_cover: float = 0.4,
    ignore_whitespace: bool = True,
    ignore_case: bool = False,
    normalize_unicode: bool = True,
) -> Dict[str, float]:
    """
    奖励（0-1，越少 n-gram 连续重复越高）：
      r1 = sigmoid( beta * (allowed_max_blocks - max_blocks) )
      r2 = exp( -alpha * coverage_ratio )
      reward = w_blocks * r1 + w_cover * r2
    """
    import math
    m = ngram_repeat_metrics(
        s, n_min, n_max,
        ignore_whitespace=ignore_whitespace,
        ignore_case=ignore_case,
        normalize_unicode=normalize_unicode,
    )
    max_blocks = m["max_blocks"]
    coverage = m["coverage_ratio"]

    r1 = 1 / (1 + math.exp(-beta * (allowed_max_blocks - max_blocks)))
    r2 = math.exp(-alpha * coverage)
    reward = w_blocks * r1 + w_cover * r2
    reward = max(0.0, min(1.0, float(reward)))

    return {
        "reward": reward,
        **m,
        "r1_blocks": r1,
        "r2_coverage": r2,
        "allowed_max_blocks": float(allowed_max_blocks),
        "beta": float(beta),
        "alpha": float(alpha),
        "w_blocks": float(w_blocks),
        "w_cover": float(w_cover),
        "n_min": float(n_min),
        "n_max": float(n_max),
    }

# ===== 示例 =====
# if __name__ == "__main__":
#     samples = [
#         "hellohello",      # n=5: "hello"×2
#         "abcabcabc",       # n=3: "abc"×3
#         "哈哈哈哈",          # n=2: "哈哈"×2；n=1: 也会有长重复，但此处只看 n>=2
#         "heeeey!!!",       # n=1 不计；此例主要是单字符拉长，对 n>=2 更宽松
#         "今天天气天气不错",   # "天气"×2
#         "abcdefg",         # 无 n-gram 连续重复
#     ]
#     for s in samples:
#         out = ngram_repeat_reward(s, n_min=2, n_max=5)
#         print(f"{s!r} -> reward={out['reward']:.3f}, max_blocks={out['max_blocks']:.0f} (n={out['max_blocks_n']:.0f}), coverage={out['coverage_ratio']:.3f}, runs={int(out['num_runs'])}")