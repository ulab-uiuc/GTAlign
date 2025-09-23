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
from nltk.translate.bleu_score import sentence_bleu
from time import sleep
import re
from recipe.gamellm.tasks.format import format_reward, extract_tag_block, response_reward_logistic, reasoning_penalty, parse_payoff_block, pareto_frontier_from_json
from openai import OpenAI

BASE_URL = "http://localhost:36485/v1"
API_KEY = "EMPTY"
MAX_RETRIES = 3
BASE_DELAY = 2

ACCURACY_PROMPT = '''You are a helpful and meticulous evaluator. Your task is to \
evaluate the *accuracy* of an AI model's answer to a target question. \
You will be given the target question, the ground truth answer, and the model's response.

Provided Information:

<|The Start of Target Question and Ground Truth Answer|>
Target Question: {problem}
Ground Truth Answer: {ground_truth}
<|The End of Target Question and Ground Truth Answer|>

<|The Start of The Model's Response|>
{completion}
<|The End of The Model's Response|>

You should determine whether the model's final response to the target question is \
factually correct and consistent with the provided ground truth.

Rating criteria (binary):
  • 1 = Correct   — the response matches the ground truth.
  • 0 = Incorrect — the response contradicts or misses the ground truth.

Your output should include a binary accuracy rating enclosed in \\boxed{{}}. Only output 0 or 1 inside the box.

Example:
thought: 'The model answer correctly matches the key fact mentioned in the ground truth.'
\\boxed{{1}}

Rules:
- Do NOT include Markdown or explanations outside of the answer.
- Use \\boxed{{}} to clearly indicate your final binary judgment (0 or 1).
- Make sure the output is unambiguous and contains only the reasoning and the boxed answer.
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
        print("未找到 boxed 数字 math-hard")
    return number if match else 0



PARETO_CONSISTENCY_PROMPT = '''You are a careful and principled evaluator.
Your task is to score whether the analyst’s explanation in <analyze> correctly identifies the model’s chosen action, compared to the ground-truth Pareto frontier.

Provided Information:

<|Ground Truth Pareto Frontier|>
# The true Pareto-optimal strategy/action label(s).
# Each strategy is written in the form "X_Y", where the model’s own action is the part after the underscore ("Y").
pareto_frontiers = {pareto_frontiers}
<|End Ground Truth Pareto Frontier|>

<|Analyst's Explanation|>
analysis_str = """{analysis_str}"""
<|End Analyst's Explanation|>

Evaluation Criteria (Action Consistency):
- Score = 1 (consistent) if:
  • The explanation explicitly identifies at least one correct action from the ground-truth frontier (the part after the underscore).
  • Extra or missing actions are tolerated, as long as at least one ground-truth action is correctly recognized.
- Score = 0.5 (partially consistent) if:
  • The explanation hints at or vaguely describes a correct action (e.g., implies clarification without naming "CQ"), OR
  • The explanation lists a superset where the correct action is overshadowed by stronger emphasis on incorrect ones.
- Score = 0 (inconsistent) if:
  • The explanation fails to mention or imply any correct ground-truth action, OR
  • It only claims incorrect actions, OR
  • It is too vague to determine any action.

Notes:
- Treat action names as equal if they are the same up to whitespace, case, or trivial formatting.
- Narrative differences are fine; correctness is judged solely by whether at least one action label after the underscore matches the ground truth.

Scoring:
→ 1 = At least one correct action clearly identified  
→ 0.5 = A correct action vaguely implied or mixed with stronger wrong claims  
→ 0 = No correct action identified  

Output Requirements:
- Provide short reasoning, explicitly referencing whether a ground-truth action was matched, partially matched, or not matched.
- Then give the rating enclosed in tags on a separate line, using <po>1</po>, <po>0.5</po>, or <po>0</po>.
- Do NOT include any explanations, Markdown, or text outside the required format.

Example (consistent):
thought: "The explanation says the model took CQ, which is one of the ground-truth actions."
<po>1</po>

Example (partially consistent):
thought: "The explanation vaguely describes asking a question (CQ) but emphasizes DA incorrectly."
<po>0.5</po>

Example (inconsistent):
thought: "The explanation only names DA, but no ground-truth actions are mentioned."
<po>0</po>
'''


def call_pareto(pareto_frontiers, solution_str, judge):
    def extract_scores_pareto(text: str):
        po_match = re.search(r"<po>(.*?)</po>", text, re.DOTALL)
        po_score = po_match.group(1).strip() if po_match else 0

        return po_score
    try:
        pareto_frontiers = ", ".join(pareto_frontiers)
        prompt = PARETO_CONSISTENCY_PROMPT.format(pareto_frontiers=pareto_frontiers, analysis_str=solution_str)
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
            po_score = extract_scores_pareto(response)
            return float(po_score)
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print("Exception: ", repr(e))
                delay = BASE_DELAY * (attempt)
                print(f"Retrying in {delay} seconds...")
                sleep(delay)
            else:
                print(f"Failed after {MAX_RETRIES} attempts. Error: {e}")

    return 0


def call_judge(solution_str, ground_truth, judge, extra_info):
    prompt = ACCURACY_PROMPT.format(problem=extra_info["single_turn_prompt"], ground_truth = ground_truth, completion=solution_str)
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
            # print("response: ", response)
            response = extract_boxed_score(response)
            return int(response)
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
        analysis_str = extract_tag_block(ori_solution_str,"analyze")
    except:
        original_reward = {
            "score": 0,
            "answer_reward": 0,
            "format_reward": 0,
            "pareto_score": 0,
            "payoff_reward": 0,
            "user_reward": 0.0,
            "llm_reward": 0.0,
            "utility": 0,
            "reward_gap": 0,
        }
        return original_reward
    try:
        pareto_frontiers = pareto_frontier_from_json(payoff_json)
    except:
        pareto_frontiers = None
    pareto_score = call_pareto(pareto_frontiers, analysis_str, judge)
    payoff_reward = 0
    for key, value in payoff_json.items():
        if value["user"] != value["LLM"]:
            payoff_reward += 1/12
        denom = max(abs(value["user"]), abs(value["LLM"]), 1e-8)  # 避免除零
        if abs(value["user"] - value["LLM"]) / denom > 0.1:
            payoff_reward += 1/12
    judge_score = call_judge(solution_str, ground_truth, judge, extra_info)
    answer_reward = float(judge_score)
    answer_reward = max(0, min(1, answer_reward))
    epsilon = 1e-4
    user_reward = epsilon + 0.5 * answer_reward + 0.2 * response_reward_logistic(solution_str, L_min=20, L_max=500, k=0.005) + 0.3 * reasoning_penalty(reasoning_str,scale=500)
    llm_reward = epsilon + 0.2 * pareto_score + 0.2 * payoff_reward + 0.2 * format_reward1 + 0.35 * answer_reward + 0.05 * response_reward_logistic(ori_solution_str, L_min=500, L_max=1000, k=0.005)
    score = (user_reward * llm_reward) ** 0.5
    original_reward = {
        "score": score,
        "answer_reward": answer_reward,
        "format_reward": format_reward1,
        "pareto_score": pareto_score,
        "payoff_reward": payoff_reward,
        "user_reward": user_reward,
        "llm_reward": llm_reward,
        "utility": score,
        "reward_gap": abs(user_reward - llm_reward),
    }
    # print(f"format_reward1: {format_reward1}")
    # print(f"answer_reward: {answer_reward}")
    return original_reward


def test():
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
    )
    out = compute_score("""<thinking>1+1=2</thinking><payoff>{
        "DQ_AQ": {
        "LLM": 4.1,
        "user": 3.5
        },
        "DQ_CQ": {
        "LLM": 3.6,
        "user": 2
        },
        "DQ_DA": {
        "LLM": 2.8,
        "user": 4.1
        },
        "VQ_AQ": {
        "LLM": 3,
        "user": 2.8
        },
        "VQ_CQ": {
        "LLM": 1,
        "user": 1.3
        },
        "VQ_DA": {
        "LLM": 1,
        "user": 1
        }
        }</payoff><analyze>1+1=2 is correct</analyze><response>1+1=\\boxed{2}</response>""", "\\boxed{2}", client, {"single_turn_prompt": "solve 1+1?"})
    print(out)

if __name__ == "__main__":
    test()