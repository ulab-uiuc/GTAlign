


# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import base64
import json
import multiprocessing
import pickle
import zlib
from nltk.translate.bleu_score import sentence_bleu
from time import sleep
import requests
import re
from recipe.gamellm.tasks.format import format_reward, extract_tag_block, response_reward_logistic, reasoning_penalty, parse_payoff_block
from openai import OpenAI

import re

def extract_scores(text: str):
    abg_match = re.search(r"<abg>(.*?)</abg>", text, re.DOTALL)
    abg_score = abg_match.group(1).strip() if abg_match else 0

    return abg_score

BASE_URL = "http://localhost:36485/v1"
API_KEY = "EMPTY"
MAX_RETRIES = 3
BASE_DELAY = 2


AMBIGUITY_PROMPT = '''You are a careful and principled evaluator. 
Your task is to evaluate an AI model's answer to a question along the single dimension of handling ambiguity.

Provided Information:

<|The Start of Target Question and Tags|>
Target Question: {question}
Ambiguity: {ambiguity}   # "ambiguous" or "non-ambiguous"
<|The End of Target Question and Tags|>

<|The Start of The Model's Response|>
{completion}
<|The End of The Model's Response|>

Evaluation Criteria for Ambiguity Handling:
- If the question is ambiguous:
  • The response should either ask clarifying questions, OR
  • Provide an answer that considers multiple possible interpretations.
- If the question is non-ambiguous:
  • The response should directly provide an effective and relevant answer.

Scoring:
→ 1 = Good ambiguity handling
→ 0 = Poor ambiguity handling

Output Requirements:
- Provide short reasoning (start with "thought:").
- Then give the binary rating enclosed in tags on a separate line.
- Do NOT include any explanations, Markdown, or text outside the required format.

Example (ambiguous question, good handling):
thought: 'The model asked a clarifying question, which is appropriate.'
<abg>1</abg>

Example (non-ambiguous question, poor handling):
thought: 'The model gave an unrelated answer instead of addressing the question directly.'
<abg>0</abg>
'''


def extract_boxed_score(text: str) -> str:
    """
    从文本中提取 \\boxed{...} 里的内容，返回字符串。
    如果没有找到，返回 None。
    """
    match = re.search(r"\\boxed\{(\d+)\}", text)
    if match:
        number = int(match.group(1))
    else:
        print("未找到 boxed 数字 abgqa")
    return number if match else 0

def call_judge(solution_str, ground_truth, judge, extra_info):
    tmp = ground_truth.split("###")
    if tmp[0] == "multipleQAs":
        ambiguity = "ambiguous"
    else:
        ambiguity = "non-ambiguous"
    standard_answer = tmp[1]
    try:
        prompt = AMBIGUITY_PROMPT.format(question=extra_info["single_turn_prompt"], ambiguity=ambiguity, completion=solution_str, standard_answer=standard_answer)
    except Exception as e:
        print(f"Error in call_judge: {e}")
        return 0
    for attempt in range(MAX_RETRIES):
        try:
            output = judge.chat.completions.create(
                model="random",
                messages=[
                        {"role": "user", "content": prompt},
                ],
                max_tokens=32768,
                temperature=0.6,
                top_p=0.95,
                extra_body={
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": True},
                }
            )
            response = output.choices[0].message.content
            abg_score = extract_scores(response)
            # print(response)
            return int(abg_score)
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print("Exception: ", repr(e))
                delay = BASE_DELAY * (attempt)
                print(f"Retrying in {delay} seconds...")
                sleep(delay)
            else:
                print(f"Failed after {MAX_RETRIES} attempts. Error: {e}")

    return 0

def compute_score(solution_str, ground_truth, judge, extra_info):

    try:
        format_reward1 = format_reward([solution_str])[0]
        ori_solution_str = solution_str
        reasoning_str = solution_str.split("<response>")[0]
        solution_str = extract_tag_block(solution_str,"response")
        payoff_matrix = extract_tag_block(ori_solution_str,"payoff")
        payoff_json = parse_payoff_block(payoff_matrix)
    except:
        original_reward = {
            "score": 0,
            "answer_reward": 0,
            "format_reward": 0,
            "user_reward": 0.0,
            "llm_reward": 0.0,
            "utility": 0,
            "reward_gap": 0,
        }
        return original_reward
    # try:
    #     pareto_frontiers = pareto_frontier_from_json(payoff_str)
    # except:
    #     pareto_frontiers = None
    # pareto_score = call_pareto(pareto_frontiers, solution_str, judge)
    payoff_reward = 0
    for key, value in payoff_json.items():
        if value["user"] != value["LLM"]:
            payoff_reward += 1/12
        denom = max(abs(value["user"]), abs(value["LLM"]), 1e-8)  # 避免除零
        if abs(value["user"] - value["LLM"]) / denom > 0.1:
            payoff_reward += 1/12
    abg_score = call_judge(solution_str, ground_truth, judge, extra_info)
    abg_score = max(0, min(1, abg_score))
    epsilon = 1e-4
    user_reward = epsilon + 0.5 * abg_score + 0.2 * response_reward_logistic(solution_str, L_min=20, L_max=500, k=0.005) + 0.3 * reasoning_penalty(reasoning_str,scale=500)
    llm_reward = epsilon + 0.2 * format_reward1 + 0.2 * payoff_reward + 0.4 * abg_score + 0.2 * response_reward_logistic(ori_solution_str, L_min=500, L_max=1000, k=0.005)
    score = (user_reward * llm_reward) ** 0.5
    # print(payoff_reward)
    original_reward = {
        "score": user_reward,
        "answer_reward": abg_score,
        "format_reward": 0.1 * format_reward1 + 0.15 * payoff_reward,
        "user_reward": user_reward,
        "llm_reward": llm_reward,
        "utility": score,
        "reward_gap": abs(user_reward - llm_reward),
    }
    # print(f"format_reward1: {format_reward1}")
    # print(f"answer_reward: {score}")
    return original_reward

def test(): 
    question="What is the outer shell of a seed called?"
    ground_truth="""singleAnswer###hull"""
    response="""<thinking>The user’s question is **DQ** (definite/clear), as it explicitly asks for the name of the outer shell of a seed. **DA** (direct answer) is optimal here, as the seed coat is the correct term, ensuring immediate clarity and correctness. **CQ** would delay resolution, risking user frustration, while **AQ** (answer + follow-up) might add unnecessary steps, especially since the question is unambiguous. DA minimizes answer correctness risk, resolves ambiguity quickly, and satisfies the user efficiently, while keeping token cost low and user effort minimal. In contrast, CQ and AQ could lead to token overuse and user dissatisfaction, particularly if the assistant overcomplicates the response.</thinking><payoff>{'VQ_DA': {'user': 1.0, 'LLM': 2.0}, 'VQ_CQ': {'user': 1.0, 'LLM': 1.0}, 'VQ_AQ': {'user': 1.0, 'LLM': 1.0}, 'DQ_DA': {'user': 3.0, 'LLM': 3.0}, 'DQ_CQ': {'user': 2.0, 'LLM': 2.0}, 'DQ_AQ': {'user': 2.0, 'LLM': 2.0}}</payoff><analyze>user position: DQ – clear goal, no missing info. Pareto optimum: DA – LLM payoff 3.0 > CQ 2.0 and AQ 2.0; user payoff 3.0 also highest. Response plan: Provide the seed coat name and no follow-up.</analyze><response>The outer shell of a seed is called the **seed coat**. It is a protective layer that surrounds the embryo and helps protect the developing seed until it can germinate. The seed coat is often tough and impermeable, preventing water loss and protecting the delicate embryo from environmental hazards.</response>"""
    extra_info = {"prompt": [{"role": "user", "content": question}], "single_turn_prompt": question}
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
    )
    out = compute_score(judge=client, solution_str=response, ground_truth=ground_truth, extra_info=extra_info)
    print(out)

if __name__ == "__main__":
    test()
