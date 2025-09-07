import re
import json
import ast

import math

import json

REQUIRED_KEYS = {"DQ_AQ", "DQ_CQ", "DQ_DA", "VQ_AQ", "VQ_CQ", "VQ_DA"}
REQUIRED_SUBKEYS = {"LLM", "user"}
BASE_URL = "http://[2605:340:cd51:4900:d982:421e:d837:4bf5]:36485/v1"
API_KEY = "EMPTY"
MAX_RETRIES = 3
BASE_DELAY = 2
# JUDGE_PARETO_PROMPT = """
# You are a judge for a multi-objective optimization problem. The problem has two objectives: ambiguity and correctness. The ambiguity objective measures the ambiguity of the model's response, and the correctness objective measures the correctness of the model's response. The model's response is given as input. Your task is to judge the model's response with respect to these two objectives.

# <|Add-Completion|><think>Add</think>"""

# def call_pareto(pareto_frontiers, solution_str, judge):
#     try:
#         prompt = JUDGE_PARETO_PROMPT.format(pareto_frontiers=pareto_frontiers, solution=solution_str)
#     except Exception as e:
#         print(f"Error in call_judge: {e}")
#         return 0
#     for attempt in range(MAX_RETRIES):
#         try:
#             output = judge.chat.completions.create(
#                 model="random",
#                 messages=[
#                         {"role": "user", "content": prompt},
#                 ],
#                 max_tokens=16384,
#             )
#             response = output.choices[0].message.content
#             abg_score, correctness_score = extract_scores(response)
#             response = (int(abg_score) + int(correctness_score)) / 2
#             return response
#         except Exception as e:
#             if attempt < MAX_RETRIES - 1:
#                 print("Exception: ", repr(e))
#                 delay = BASE_DELAY * (attempt)
#                 print(f"Retrying in {delay} seconds...")
#                 sleep(delay)
#             else:
#                 print(f"Failed after {MAX_RETRIES} attempts. Error: {e}")

#     return 0

def extract_tag_block(text: str, tag: str) -> str:
    """提取 <tag>...</tag> 的原始内容（不含外层标签）。若无则抛错。"""
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"Missing <{tag}> block")
    return m.group(1).strip()

def jsonish_to_json(s: str) -> str:
    """
    将 Python 风格字典/列表字符串（单引号、True/False/None）转为合法 JSON 字符串。
    优先尝试 json 解析；失败则走 ast.literal_eval 再 dumps。
    """
    s_stripped = s.strip()
    # 1) 直接是 JSON？
    try:
        json.loads(s_stripped)
        return s_stripped
    except Exception:
        pass

    # 2) 修正常见“弯引号”
    s_fixed = (s_stripped
               .replace("“","\"").replace("”","\"")
               .replace("‘","'").replace("’","'"))
    # 3) 用 ast.literal_eval 把 Python 字面量解析成对象
    try:
        py_obj = ast.literal_eval(s_fixed)
    except Exception as e:
        raise ValueError(f"Cannot parse payoff as JSON/Python literal: {e}")

    # 4) 再转为合法 JSON
    return json.dumps(py_obj, ensure_ascii=False)

def parse_payoff_block(text: str) -> dict:
    """
    从整段文本中解析 payoff 区块，并严格验证结构：
    - 是包含 6 个固定策略键的 dict
    - 每个值是包含 "LLM" 和 "user" 键的子 dict，值为 float
    """
    json_str = jsonish_to_json(text)
    data = json.loads(json_str)

    # Step 1: 校验键集合
    keys = set(data.keys())
    if keys != REQUIRED_KEYS:
        missing = REQUIRED_KEYS - keys
        extra   = keys - REQUIRED_KEYS
        msg = []
        if missing: msg.append(f"missing: {sorted(missing)}")
        if extra:   msg.append(f"extra: {sorted(extra)}")
        raise ValueError("Payoff keys mismatch: " + "; ".join(msg))

    # Step 2: 校验每个子结构
    for k, v in data.items():
        if not isinstance(v, dict):
            raise ValueError(f"Payoff value for '{k}' must be a dict, got {type(v)}")

        sub_keys = set(v.keys())
        if sub_keys != REQUIRED_SUBKEYS:
            missing = REQUIRED_SUBKEYS - sub_keys
            extra   = sub_keys - REQUIRED_SUBKEYS
            msg = []
            if missing: msg.append(f"missing subkeys: {sorted(missing)}")
            if extra:   msg.append(f"extra subkeys: {sorted(extra)}")
            raise ValueError(f"Subkeys mismatch for '{k}': " + "; ".join(msg))

        for sk, val in v.items():
            if not isinstance(val, (float, int)):
                raise ValueError(f"Value of '{sk}' in '{k}' must be float or int, got {type(val)}")

    return data


import json
from math import isclose

def is_dominated(a, b, eps=1e-9):
    """
    判断点 a 是否被点 b 支配（最大化问题）。
    a, b: (llm, user) 元组
    若 b 在两维都 >= a，且至少一维 >（容忍 eps），则 a 被支配。
    """
    llm_a, user_a = a
    llm_b, user_b = b
    ge_all = (llm_b > llm_a + eps or isclose(llm_b, llm_a, abs_tol=eps)) and \
             (user_b > user_a + eps or isclose(user_b, user_a, abs_tol=eps))
    gt_any = (llm_b > llm_a + eps) or (user_b > user_a + eps)
    return ge_all and gt_any

def pareto_frontier_from_json(json_str, eps=1e-9):
    data = json.loads(json_str)
    # 将字典转成列表 [(name, (llm, user)), ...]
    items = [(k, (v["LLM"], v["user"])) for k, v in data.items()]
    frontier = []
    for i, (name_i, pt_i) in enumerate(items):
        dominated = False
        for j, (name_j, pt_j) in enumerate(items):
            if i == j:
                continue
            if is_dominated(pt_i, pt_j, eps=eps):
                dominated = True
                break
        if not dominated:
            frontier.append((name_i, pt_i))
    # 常用：按 LLM 分数升序（或改 key=lambda x: x[1][1] 按 user 升序）
    frontier.sort(key=lambda x: (x[1][0], x[1][1]))
    return frontier

def format_reward(completions, **kwargs):
    pattern = (
        r"^<thinking>.*?</thinking>\s*"
        r"<payoff>.*?</payoff>\s*"
        r"<analyze>.*?</analyze>\s*"
        r"<response>.*?</response>$"
    )

    rewards = []

    for i, content in enumerate(completions):
        # Step 1: 格式检查
        match = re.match(pattern, content, re.DOTALL | re.MULTILINE)
        if not match:
            rewards.append(0.0)
            continue

        # Step 2: 提取 <payoff> 块
        payoff_str = extract_tag_block(content, "payoff")

        # Step 3: 尝试 JSON 解析
        payoff_json = None
        if payoff_str:
            try:
                payoff_json = parse_payoff_block(payoff_str)
                rewards.append(1.0)
                continue
            except Exception as e:
                rewards.append(0.0)
                continue
        else:
            # <payoff></payoff> 是空的
            rewards.append(0.0)

    return rewards

# print(format_reward(["""<thinking>123</thinking><payoff>{
#   "DQ_AQ": {"LLM": 2.2, "user": 1.9},
#   "DQ_CQ": {"LLM": 3.1, "user": 3.5},
#   "DQ_DA": {"LLM": 4.2, "user": 4.3},
#   "VQ_AQ": {"LLM": 2.0, "user": 2.1},
#   "VQ_CQ": {"LLM": 1.3, "user": 1.2},
#   "VQ_DA": {"LLM": 2.0, "user": 1.8}
# }</payoff><analyze>123</analyze><response>123</response>"""]))


# Overlength penalty for the argument part
# def overlength_penalty_reward(completions, config, tokenizer, **kwargs):
#     completion_contents = completions
#     completion_contents = [_.strip() for _ in completion_contents]
#     max_arg_length = config.data.max_arg_length
#     results = []
    
#     for content in completion_contents:
#         argument_text = extract_answer(content, 'argument')
#         if argument_text is None:
#             results.append(0.0)
#             continue
#         tokens = tokenizer.encode(argument_text, add_special_tokens=False)
#         token_count = len(tokens)
#         if token_count <= max_arg_length:
#             results.append(0.0)
#         else:
#             overrun_ratio = (token_count - max_arg_length) / max_arg_length
#             penalty = min(overrun_ratio * 2, 1.0) * -1.0  # reaches maximum penalty of -1.0 when overlength for 50% or more
#             results.append(penalty)
    
#     return results


# 暂时不用这个
# def make_repetition_penalty_reward(config=None):
#     # ngram_size = config.trainer.rep.ngram_size
#     # threshold = config.trainer.rep.threshold
#     # max_penalty = -1
#     ngram_size = 3
#     max_penalty = -1
#     threshold = 0.2
#     if max_penalty > 0:
#         raise ValueError(f"max_penalty {max_penalty} should not be positive")

#     def zipngram(text: str, ngram_size: int):
#         words = text.lower().split()
#         return zip(*[words[i:] for i in range(ngram_size)])

#     def repetition_penalty_reward(completions: List[str], turns: List[List[str]], **kwargs) -> List[float]:
#         contents = completions
#         rewards = []
#         for completion, turn_history in zip(contents, turns):
#             if completion == "":
#                 rewards.append(0.0)
#                 continue
#             if len(completion.split()) < ngram_size:
#                 rewards.append(0.0)
#                 continue
#             curr_ngrams = set()
#             curr_total = 0
#             for ng in zipngram(completion, ngram_size):
#                 curr_ngrams.add(ng)
#                 curr_total += 1
            
#             curr_scaling = 1 - len(curr_ngrams) / curr_total if curr_total > 0 else 0

#             history_ngrams = set()
#             history_total = 0
#             for prev_text in turn_history[:-2][::2]:
#                 for ng in zipngram(prev_text, ngram_size):
#                     history_ngrams.add(ng)
#                     history_total += 1
#             if history_total > 0:
#                 overlap_ngrams = curr_ngrams.intersection(history_ngrams)
#                 history_scaling = len(overlap_ngrams) / len(curr_ngrams) if curr_ngrams else 0
#             else:
#                 history_scaling = 0

#             scaling = max(curr_scaling, history_scaling)
#             reward = scaling * max_penalty
#             reward = min(0.0, reward + threshold)
#             rewards.append(reward)
            
#         return rewards

#     return repetition_penalty_reward

import re
from itertools import combinations
from typing import List, Tuple, Dict, Any, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _split_sentences(text: str) -> List[str]:
    """中英混合的轻量分句。"""
    parts = re.split(r"[。．\.！？!?；;:\n]+", text)
    return [p.strip() for p in parts if p.strip()]


def _windows(text: str, window_chars: int = 60, step: int = 40) -> List[str]:
    """
    当文本没有明显句号时，可用定长字符窗口切片。
    window_chars: 窗口长度（字符）
    step: 滑动步长（字符）
    """
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    if len(text) <= window_chars:
        return [text]
    return [text[i:i+window_chars] for i in range(0, len(text)-window_chars+1, step)]


def semantic_redundancy_score_tfidf(
    text: str,
    use_windows_if_few_sentences: bool = True,
    window_chars: int = 60,
    step: int = 40,
    ngram_range: Tuple[int, int] = (1, 2),
    min_docs: int = 2,
    top_p: float = 0.3,
    repetition_threshold: float = 0.8,
) -> Dict[str, Any]:
    """
    计算字符串内部语义重复度（0~1）。返回字典包含分数、是否重复、最相似片段等信息。

    参数说明：
    - use_windows_if_few_sentences: 分句后片段太少时，改用滑动窗口。
    - window_chars/step: 滑动窗口大小与步长（字符级）。
    - ngram_range: TF-IDF 的 n-gram 范围，(1,2) 更敏感；(1,1) 更保守；(1,3) 更激进。
    - min_docs: 至少需要多少个片段才进行计算。
    - top_p: 只聚合最相似的前 top_p 比例的片段对，减少无关片段稀释（0.2~0.5 常用）。
    - repetition_threshold: 单对片段余弦相似度超过该阈值，即判为“明显重复”。

    返回：
    {
      "score": float,                 # 0~1，越大越重复
      "is_repetitive": bool,          # 是否存在明显重复（基于 repetition_threshold）
      "max_pair": (i, j, score),      # 最相似的片段对及其得分
      "high_pairs": [((i, j), s), ...], # 所有超过阈值的片段对
      "chunks": [ ... ]               # 参与计算的片段文本
    }
    """
    # 1) 分句或滑窗切片
    chunks = _split_sentences(text)
    if len(chunks) < min_docs and use_windows_if_few_sentences:
        chunks = _windows(text, window_chars=window_chars, step=step)

    # 过滤过短片段
    chunks = [c for c in chunks if len(c) >= 3]
    if len(chunks) < min_docs:
        return {
            "score": 0.0,
            "is_repetitive": False,
            "max_pair": None,
            "high_pairs": [],
            "chunks": chunks,
        }

    # 2) TF-IDF 向量化
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, lowercase=True, token_pattern=r"(?u)\b\w+\b")
    X = vectorizer.fit_transform(chunks)

    # 3) 计算两两余弦相似度（只取上三角）
    sim_mat = cosine_similarity(X)
    sims = []
    pairs = []
    n = len(chunks)
    for i, j in combinations(range(n), 2):
        s = float(sim_mat[i, j])
        sims.append(s)
        pairs.append((i, j, s))

    if not sims:
        return {
            "score": 0.0,
            "is_repetitive": False,
            "max_pair": None,
            "high_pairs": [],
            "chunks": chunks,
        }

    # 4) 聚合：只平均最相似的前 top_p 比例
    pairs_sorted = sorted(pairs, key=lambda x: x[2], reverse=True)
    k = max(1, int(len(pairs_sorted) * top_p))
    top_pairs = pairs_sorted[:k]
    score = sum(s for _, _, s in top_pairs) / k

    # 5) 阈值判断与输出
    high_pairs = [((i, j), s) for i, j, s in pairs_sorted if s >= repetition_threshold]
    max_pair = pairs_sorted[0] if pairs_sorted else None

    return {
        "score": round(score, 4),
        "is_repetitive": len(high_pairs) > 0,
        "max_pair": max_pair,          # (i, j, max_sim)
        "high_pairs": high_pairs,      # [((i, j), s), ...]
        "chunks": chunks,
    }

def reasoning_penalty(text: str, scale: float = 200):
    """
    返回 [0,1] 的分数。短文本分数接近 1，长文本分数接近 0。
    
    参数:
    - text: 输入字符串
    - scale: 控制衰减速度，越大越平缓
    """
    length = len(text)
    score = math.exp(-length / scale)
    return score

def response_reward_logistic(text: str, L_min=100, L_max=2000, k=0.005):
    length = len(text.split())
    left = 1 / (1 + math.exp(k * (L_min - length)))
    right = 1 / (1 + math.exp(k * (length - L_max)))
    return left * right

if __name__ == "__main__":
    # ===== 示例：用你的数据 =====
    json_input = '''{
        "DQ_AQ": {"LLM": 4.1, "user": 3.5},
        "DQ_CQ": {"LLM": 3.6, "user": 2},
        "DQ_DA": {"LLM": 2.8, "user": 4.1},
        "VQ_AQ": {"LLM": 3, "user": 2.8},
        "VQ_CQ": {"LLM": 1, "user": 1.3},
        "VQ_DA": {"LLM": 1, "user": 1}
    }'''

    frontier = pareto_frontier_from_json(json_input)
    print("Pareto frontier (max LLM & user):")
    for name, (llm, usr) in frontier:
        print(f"{name}: LLM={llm}, user={usr}")

    text = (
        "我非常喜欢机器学习，这个领域让我着迷。"
        "机器学习相关的研究我也很感兴趣。"
        "顺便说一下，昨晚我做了咖喱饭。"
        "其实我真的很喜欢机器学习。"
    )
    result = semantic_redundancy_score_tfidf(
        "我非常喜欢机器学习，这个领域让我着迷。机器学习相关的研究我也很感兴趣。其实我真的很喜欢机器学习。",
        ngram_range=(1, 2),
        top_p=0.3,
        repetition_threshold=0.75
    )
    print(result)
    # 你也可以只取最终分数：result["score"]


