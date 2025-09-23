# pip install openai==1.*
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List
import json
import logging

from openai import OpenAI
from recipe.gamellm.tasks.format import (
    format_reward,
    extract_tag_block,
    response_reward_logistic,
    reasoning_penalty,
    parse_payoff_block,
    pareto_frontier_from_json,
)
import random

# -----------------------------
# 配置 & 常量
# -----------------------------
DEFAULT_BASE_URL = "http://127.0.0.1:36486/v1"
DEFAULT_API_KEY = "EMPTY"

GAME_THEORY_PREFIX = (
    "Now you need to answer the user's most recent question. Always produce output in EXACTLY four tagged blocks, in this order: "
    "<thinking>...</thinking><payoff>...</payoff><analyze>...</analyze><response>...</response>. "
    "Do not include anything outside these tags. Do not use markdown, code fences, or extra commentary."
    "Inside <thinking>...</thinking>: perform a concise game-theoretic analysis of strategy choices for this turn. "
    "Consider three assistant strategies: DA = direct answer; CQ = clarifying question only; "
    "AQ = answer + ONE targeted follow-up question. "
    "Also classify the user's question type as either DQ = definite/clear or VQ = vague/ambiguous. "
    "Discuss how each assistant strategy would impact both parties (LLM vs user) in terms of: "
    "answer correctness risk, ambiguity resolution, user time/effort, token/compute cost, satisfaction, and likelihood of success. "
    "Keep this analysis brief but concrete (2-5 sentences). No numbers here yet."
    "Inside <payoff>...</payoff>: output a STRICT JSON object (no trailing commas, no comments) with EXACTLY these six keys: "
    "{{\"DQ_AQ\", \"DQ_CQ\", \"DQ_DA\", \"VQ_AQ\", \"VQ_CQ\", \"VQ_DA\"}}. "
    "Each key maps to an object with two float fields: {{\"LLM\": <float>, \"user\": <float>}}. "
    "Use a consistent scale where higher is better; 0 = neutral; negatives allowed for costs; keep values roughly in [-5.0, 5.0]. "
    "Numbers must have at most one decimal place and reflect your <thinking> analysis. Example shape (values are examples only): "
    "{{ \"DQ_AQ\": {{ \"LLM\": 2.2, \"user\": 1.9 }}, \"DQ_CQ\": {{ \"LLM\": 3.1, \"user\": 3.5 }}, "
    "\"DQ_DA\": {{ \"LLM\": 4.2, \"user\": 4.3 }}, \"VQ_AQ\": {{ \"LLM\": 2.0, \"user\": 2.1 }}, "
    "\"VQ_CQ\": {{ \"LLM\": 1.3, \"user\": 1.2 }}, \"VQ_DA\": {{ \"LLM\": 2.0, \"user\": 1.8 }} }}."
    "Inside <analyze>...</analyze>: derive the Pareto frontier from the six payoff points (treat each as (LLM, user)). "
    "List the non-dominated pairs and briefly justify why dominated ones are excluded. "
    "Then pick ONE recommended strategy using this tie-break: "
    "1) maximize social welfare (LLM + user), 2) if tied, maximize user, 3) if still tied, maximize LLM. "
    "Name the chosen key (e.g., DQ_DA) and justify in 1-3 sentences."
    "Inside <response>...</response>: implement the chosen strategy for the user's actual question. "
    "If you chose DA, give a concise, direct answer only. "
    "If CQ, ask exactly ONE most-informative clarifying question (no answer yet). "
    "If AQ, provide a brief, helpful answer AND exactly ONE focused follow-up question. "
    "Do not mention payoffs or the analysis tags; just act accordingly here."
    "Output policy: "
    "- Absolutely no text outside the four tags. "
    "- Keys and field names must match EXACTLY. "
    "- Keep <thinking> short and non-redundant; keep <analyze> focused."
    "Question: {question}"
)


from typing import Dict, List, Tuple

def pareto_optimal_keys(transformed_payoff: Dict[str, Dict[str, float]], eps: float = 0.0) -> List[str]:
    """
    返回所有帕累托最优的 joint strategy key（如 'DQ_DA'）。
    支持用 eps 做浮点宽松比较：a >= b 视为 (a > b - eps)。
    """
    keys = list(transformed_payoff.keys())
    efficient = []
    for k in keys:
        u, l = transformed_payoff[k]["user"], transformed_payoff[k]["LLM"]
        dominated = False
        for j in keys:
            if j == k: 
                continue
            uj, lj = transformed_payoff[j]["user"], transformed_payoff[j]["LLM"]
            # j 支配 k：两个维度都不差且至少一个严格更好（带 eps）
            if uj*lj > u* l:
                dominated = True
                break
        if not dominated:
            efficient.append(k)
    return efficient

def get_action_analysis(action: str) -> str:
    """
    根据助手行动类型，返回对应的说明文本。
    行动类型:
        - "DA": Direct Answer
        - "CQ": Clarifying Question
        - "AQ": Answer + Follow-up Question
    """
    if action == "DA":
        return "Provide a direct answer to the user's question."
    elif action == "CQ":
        return "Ask a clarifying question to better understand the user's intent."
    elif action == "AQ":
        # return "Give an answer first, then give the user some followup questions to continue the conversation."
        return "Provide a clear and helpful answer first, then encourage the user to continue the conversation by suggesting possible directions to explore."
    else:
        return "Unknown action."
    
def pareto_optimal_assistant_actions(transformed_payoff: Dict[str, Dict[str, float]], eps: float = 0.0) -> List[str]:
    """
    返回帕累托最优组合中，'_' 后面的助手行动（去重，按出现顺序）。
    例如 'DQ_DA' -> 'DA'
    """
    seen = set()
    actions = []
    for k in pareto_optimal_keys(transformed_payoff, eps=eps):
        # 拆分并取 '_' 后的动作
        try:
            assistant_act = k.split("_", 1)[1]
        except IndexError:
            # 若格式异常则跳过或可改为 raise
            continue
        if assistant_act not in seen:
            seen.add(assistant_act)
            actions.append(assistant_act)
    return actions



@dataclass
class LLMConfig:
    model: str = "default"
    base_url: str = DEFAULT_BASE_URL
    api_key: str = DEFAULT_API_KEY
    temperature: float = 0.2
    max_tokens: int = 2048
    stop_on_payoff: str = "</payoff>"  # 第一次调用的停靠点


class GameLLMOrchestrator:
    """
    将“截断生成 -> 中间处理 -> 续写生成”封装为易扩展的流水线。
    """

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self.client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
        self.log = logging.getLogger(self.__class__.__name__)

    # ---------- 提示词 ----------
    @staticmethod
    def build_prompt(question: str) -> str:
        return GAME_THEORY_PREFIX.format(question=question)

    # ---------- LLM 基础调用 ----------
    def _chat(self, messages: List[Dict[str, str]], *, stop: Optional[List[str]] = None) -> Tuple[str, str]:
        """
        统一封装 chat.completions.create，返回 (content, finish_reason)
        """
        resp = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=messages,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            stop=stop,
            stream=False,
        )
        choice = resp.choices[0]
        return (choice.message.content or ""), (choice.finish_reason or "")

    # ---------- 第一次请求：在 </payoff> 截断 ----------
    def first_stage_partial(
        self, user_prompt: str
    ) -> Tuple[str, str, Dict[str, Dict[str, float]], List[Tuple[float, float]]]:
        """
        发送第一次请求，让模型在 </payoff> 停止。
        返回：
          - partial_text: 模型到 </payoff> 为止的文本（不含停靠 token）
          - payoff_block: 解析出的 <payoff> 内容原文（补齐闭合）
          - payoff_json: 解析后的 JSON（六个键）
          - pareto_points: 六个 (LLM, user) 对
        """
        messages = [{"role": "user", "content": user_prompt}]
        partial_text, finish_reason = self._chat(messages, stop=[self.cfg.stop_on_payoff])

        if finish_reason != "stop":
            self.log.warning("first_stage_partial finish_reason=%s, 可能未命中停靠点", finish_reason)

        # 补齐 </payoff> 用于解析
        payoff_block = extract_tag_block(partial_text + "</payoff>", "payoff")
        payoff_json = parse_payoff_block(payoff_block)

        # Pareto points 仅用于可视化/校验
        pareto_points = []
        for key, v in payoff_json.items():
            pareto_points.append((float(v["LLM"]), float(v["user"])))

        return partial_text, payoff_block, payoff_json, pareto_points

    # ---------- 中间处理：自定义变换/校验 ----------
    @staticmethod
    def transform_payoffs(
        payoff_json: Dict[str, Dict[str, float]], *, scale_llm: float = 1.0, scale_user: float = 1.0
    ) -> Dict[str, Dict[str, float]]:
        """
        对 payoff 做你想要的变换（这里示例为 LLM/user 都乘以 -2）。
        你也可以在这里：
          - 做边界裁剪/合法性校验
          - 计算指标/记录日志
        """
        out = {}
        for k, v in payoff_json.items():
            if scale_llm != 0:
                if k.endswith("_DA"):
                    out[k] = {
                        "LLM": round(float(v["LLM"]) + scale_llm, 1),
                        "user": round(float(v["user"]), 1),
                    }
                elif k.endswith("_AQ"):
                    out[k] = {
                        "LLM": round(float(v["LLM"]) + scale_llm, 1),
                        "user": round(float(v["user"]) + 1, 1),
                    }
                elif k.endswith("_CQ"):
                    out[k] = {
                        "LLM": round(float(v["LLM"]) + scale_llm, 1),
                        "user": round(float(v["user"]) + 0.5, 1),
                    }
            elif scale_user != 0:
                if k.endswith("_DA"):
                    out[k] = {
                        "LLM": round(float(v["LLM"]) + 1, 1),
                        "user": round(float(v["user"]) + scale_user, 1),
                    }
                elif k.endswith("_AQ"):
                    out[k] = {
                        "LLM": round(float(v["LLM"]) + 0.5, 1),
                        "user": round(float(v["user"]) + scale_user, 1),
                    }
                elif k.endswith("_CQ"):
                    out[k] = {
                        "LLM": round(float(v["LLM"]), 1),
                        "user": round(float(v["user"]) + scale_user, 1),
                    }
            else:
                out[k] = {
                        "LLM": round(float(v["LLM"]), 1),
                        "user": round(float(v["user"]), 1),
                    }
        return out

    @staticmethod
    def build_assistant_patch(
        thinking_text: str, transformed_payoff: Dict[str, Dict[str, float]]
    ) -> str:
        """
        构造发回给模型的“助手上一轮输出”。注意需要包含完整 <thinking> 和 <payoff>。
        """
        feasible_set = pareto_optimal_assistant_actions(transformed_payoff)
        # 筛选 Pareto 最优动作
        action = random.choice(feasible_set)
        analysis = get_action_analysis(action)
        print("post feasible_set: ", feasible_set)
        print(f"action: {action}, analysis: {analysis}")
        json_str = json.dumps(transformed_payoff, ensure_ascii=False)
        # return f"<thinking>{analysis}</thinking><payoff>{json_str}</payoff><analyze>{analysis}</analyze>"
        return analysis

    # ---------- 第二次请求：基于“历史+补丁”续写 ----------
    def second_stage_continue(self, original_user_prompt: str, assistant_patch: str) -> str:
        """
        把（用户原始 prompt + 助手补丁）一起发回，促使模型从 </payoff> 之后继续生成。
        """
        messages = [
            {"role": "user", "content": original_user_prompt + assistant_patch},
            # {"role": "assistant", "content": assistant_patch},
            # 你也可以在这里附带“处理说明”作为用户消息，引导续写：
            # {"role": "user", "content": "已验证收益矩阵，请从 </payoff> 之后继续生成剩余内容。"}
        ]
        content, _ = self._chat(messages, stop=None)
        return content

    # ---------- 一键式执行 ----------
    def run(
        self,
        question: str,
        *,
        thinking_hint: str = "we should find out best response strategies based on the payoff matrix.",
        scale_llm: float = -2.0,
        scale_user: float = -2.0,
        return_debug: bool = False,
    ) -> Dict[str, Any]:
        """
        完整流程：
          1) 构造提示词
          2) 第一次生成到 </payoff>
          3) 解析 + 变换 payoff
          4) 构造“助手补丁”，作为历史消息发回
          5) 续写生成
        """
        user_prompt = self.build_prompt(question)

        partial_text, payoff_block, payoff_json, pareto_points = self.first_stage_partial(user_prompt)

        # 你也可以在这里做自定义校验，例如检查键是否齐全
        expected_keys = {"DQ_AQ", "DQ_CQ", "DQ_DA", "VQ_AQ", "VQ_CQ", "VQ_DA"}
        if set(payoff_json.keys()) != expected_keys:
            self.log.warning("payoff keys 异常：%s", list(payoff_json.keys()))
        # thinking_hint = extract_tag_block(partial_text, "thinking")
        transformed = self.transform_payoffs(payoff_json, scale_llm=scale_llm, scale_user=scale_user)
        feasible_sets = pareto_optimal_assistant_actions(payoff_json)
        print(f"pre feasible_sets: {feasible_sets}")
        assistant_patch = self.build_assistant_patch(thinking_hint, transformed)
        continuation = self.second_stage_continue(question, assistant_patch)

        result = {
            "partial_text": partial_text,
            "payoff_block": payoff_block,
            "payoff_json_raw": payoff_json,
            "payoff_json_transformed": transformed,
            "pareto_points": pareto_points,
            "assistant_patch": assistant_patch,
            "continuation": continuation,
        }
        return result if return_debug else {"continuation": continuation}


# -----------------------------
# 示例 main：直接运行脚本
# -----------------------------
def main():
    logging.basicConfig(level=logging.INFO)

    cfg = LLMConfig(
        model="default",
        base_url=DEFAULT_BASE_URL,
        api_key=DEFAULT_API_KEY,
        temperature=1,
        max_tokens=2048,
        stop_on_payoff="</payoff>",
    )
    orchestrator = GameLLMOrchestrator(cfg)

    question = "When did dinosaurs first appear?"
    out = orchestrator.run(
        question=question,
        thinking_hint="we should find out best response strategies based on the payoff matrix.",
        scale_llm=0,
        scale_user=-0.5,
        return_debug=True,  # 调试时设 True，产出更多中间结果
    )

    print("=== PARTIAL (to </payoff>) ===")
    print(out["partial_text"])
    print("\n=== RAW PAYOFF JSON ===")
    print(json.dumps(out["payoff_json_raw"], indent=2, ensure_ascii=False))
    print("\n=== TRANSFORMED PAYOFF JSON ===")
    print(json.dumps(out["payoff_json_transformed"], indent=2, ensure_ascii=False))
    print("\n=== CONTINUATION (after </payoff>) ===")
    print(out["continuation"])


if __name__ == "__main__":
    main()