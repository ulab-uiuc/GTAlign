# pip install openai==1.*  （或你已在用的版本）
from openai import OpenAI
from recipe.gamellm.tasks.format import format_reward, extract_tag_block, response_reward_logistic, reasoning_penalty, parse_payoff_block, pareto_frontier_from_json
import json

client = OpenAI(base_url="http://127.0.0.1:36486/v1", api_key="EMPTY")

GAME_THEORY_USER_CONTENT_PREFIX = (
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

user_message = GAME_THEORY_USER_CONTENT_PREFIX.format(question="please write an article on efficient llm attention for me.")
# ===== 1) 第一次请求：遇到 </payoff> 立即停 =====
resp1 = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "user", "content": user_message},
    ],
    stop=["</payoff>"],        # 关键：让模型遇到该序列就停
    temperature=0.2,
    max_tokens=2048,
    stream=False               # 也可以 True 用流式处理
)

partial_out = resp1.choices[0].message.content or ""
finish_reason = resp1.choices[0].finish_reason
print("=== PARTIAL ===")
print(partial_out)
print("finish_reason:", finish_reason)   # 一般为 "stop"

# 你的处理：例如解析 <analysis> 内容、保存 payoff、做验证等
processed_note = "（示例）我已验证收益矩阵，继续生成结论与后续步骤。"


payoff_matrix = extract_tag_block(partial_out + "</payoff>","payoff")
payoff_json = parse_payoff_block(payoff_matrix)
pareto_frontiers = pareto_frontier_from_json(payoff_json)
for key, value in payoff_json.items():
    value["LLM"] = -2 * value["LLM"]
    value["user"] = -2 * value["user"]

json_str = json.dumps(payoff_json)
print(f"{json_str=}")

partial_out_new = "<thinking>we should find out best response strategies based on the payoff matrix.</thinking>" + f"<payoff>{json_str}</payoff>"

print(f"{partial_out_new=}")
# ===== 2) 第二次请求：在你的处理后继续生成 =====
# 做法：把历史对话 + 第一次助手的“部分输出”一并发回，再追加你的新输入
messages = [
    {"role": "user", "content": GAME_THEORY_USER_CONTENT_PREFIX.format(question="please write an article on efficient llm attention for me.")},
    {"role": "assistant", "content": partial_out_new},
    # {"role": "user", "content": f"{processed_note}\n请从 </payoff> 之后继续生成剩余内容。"}
]


resp2 = client.chat.completions.create(
    model="default",
    messages=messages,
    # 这次通常不再设置 stop，除非你还有别的停靠点
    temperature=0.2,
    max_tokens=2048,
    stream=False
)

continuation = resp2.choices[0].message.content or ""
print("=== CONTINUATION ===")
print(continuation)