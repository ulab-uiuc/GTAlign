


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
from recipe.gamellm.tasks.format import format_reward, extract_tag_block, response_reward_logistic, reasoning_penalty, parse_payoff_block, pareto_frontier_from_json
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
            print(f"Pareto Consistency Response: {response}")
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
    tmp = ground_truth.split("###")
    if tmp[0] == "multipleQAs":
        ambiguity = "ambiguous"
    else:
        ambiguity = "non-ambiguous"
    standard_answer = tmp[1]
    try:
        prompt = AMBIGUITY_PROMPT.format(question=extra_info["single_turn_prompt"], ambiguity=ambiguity, completion=solution_str)
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
    abg_score = call_judge(solution_str, ground_truth, judge, extra_info)
    abg_score = max(0, min(1, abg_score))
    epsilon = 1e-4
    user_reward = epsilon + 0.5 * abg_score + 0.2 * response_reward_logistic(solution_str, L_min=20, L_max=500, k=0.005) + 0.3 * reasoning_penalty(reasoning_str,scale=500)
    llm_reward = epsilon + 0.2 * pareto_score + 0.2 * payoff_reward + 0.2 * format_reward1 + 0.35 * abg_score + 0.05 * response_reward_logistic(ori_solution_str, L_min=500, L_max=1000, k=0.005)
    score = (user_reward * llm_reward) ** 0.5
    original_reward = {
        "score": score,
        "answer_reward": abg_score,
        "format_reward": format_reward1,
        "pareto_score": pareto_score,
        "payoff_reward": payoff_reward,
        "user_reward": user_reward,
        "llm_reward": llm_reward,
        "utility": score,
        "reward_gap": abs(user_reward - llm_reward),
    }
    return original_reward

def test(): 
    question="What is the outer shell of a seed called?"
    ground_truth="""singleAnswer###hull"""
    response = """
    <thinking>\n- Question type: DQ (Definite/Clear)\n- DA (Direct Answer): Owner is First Bank of Nigeria Limited.\n- CQ (Clarifying Question Only): No additional question needed as the question is clear.\n- AQ (Answer + Follow-up Question): Provide the answer and ask for confirmation.\n</thinking>\n<payoff>\n{\"DQ_AQ\": {\"LLM\": 2.0, \"user\": 2.1}, \"DQ_CQ\": {\"LLM\": 3.1, \"user\": 3.5}, \"DQ_DA\": {\"LLM\": 4.2, \"user\": 4.3}, \"VQ_AQ\": {\"LLM\": 2.0, \"user\": 2.1}, \"VQ_CQ\": {\"LLM\": 1.3, \"user\": 1.2}, \"VQ_DA\": {\"LLM\": 2.0, \"user\": 1.8}}\n</payoff>\n<analyze>\n- DA: Directly answers the question, ensuring correctness but minimal user effort.\n- CQ: Provides no answer, high user effort and risk of ambiguity.\n- AQ: Provides the answer and a follow-up for confirmation, balancing correctness and user effort.\n- VQ: Provides no answer, high user effort and risk of ambiguity.\n- DA and AQ are non-dominated, but DA is slightly better for user satisfaction.\n</analyze>\n<response>\nAQ: The owner of the first bank in Nigeria is First Bank of Nigeria Limited. Is this correct?\n</response>
    """
    # response="""<thinking>The user’s question is **DQ** (definite/clear), as it explicitly asks for the name of the outer shell of a seed. **DA** (direct answer) is optimal here, as the seed coat is the correct term, ensuring immediate clarity and correctness. **CQ** would delay resolution, risking user frustration, while **AQ** (answer + follow-up) might add unnecessary steps, especially since the question is unambiguous. DA minimizes answer correctness risk, resolves ambiguity quickly, and satisfies the user efficiently, while keeping token cost low and user effort minimal. In contrast, CQ and AQ could lead to token overuse and user dissatisfaction, particularly if the assistant overcomplicates the response.</thinking><payoff>{'VQ_DA': {'user': 1.0, 'LLM': 2.0}, 'VQ_CQ': {'user': 1.0, 'LLM': 1.0}, 'VQ_AQ': {'user': 1.0, 'LLM': 1.0}, 'DQ_DA': {'user': 3.0, 'LLM': 3.0}, 'DQ_CQ': {'user': 2.0, 'LLM': 2.0}, 'DQ_AQ': {'user': 2.0, 'LLM': 2.0}}</payoff><analyze>user position: DQ – clear goal, no missing info. Pareto optimum: DA – LLM payoff 3.0 > CQ 2.0 and AQ 2.0; user payoff 3.0 also highest. Response plan: Provide the seed coat name and no follow-up.</analyze><response>The outer shell of a seed is called the **seed coat**. It is a protective layer that surrounds the embryo and helps protect the developing seed until it can germinate. The seed coat is often tough and impermeable, preventing water loss and protecting the delicate embryo from environmental hazards.</response>"""
    extra_info = {"prompt": [{"role": "user", "content": question}], "single_turn_prompt": question}
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
    )
    out = compute_score(judge=client, solution_str=response, ground_truth=ground_truth, extra_info=extra_info)
    print(out)

if __name__ == "__main__":
    test()
