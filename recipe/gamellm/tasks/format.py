import re
import json
import ast


import json

REQUIRED_KEYS = {"DQ_AQ", "DQ_CQ", "DQ_DA", "VQ_AQ", "VQ_CQ", "VQ_DA"}
REQUIRED_SUBKEYS = {"LLM", "user"}

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
                rewards.append(0.5)
                continue
        else:
            # <payoff></payoff> 是空的
            rewards.append(0.5)

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

