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
from recipe.gamellm.tasks.format import format_reward, extract_tag_block, response_reward_logistic, reasoning_penalty
from openai import OpenAI

BASE_URL = "http://localhost:30003/v1"
API_KEY = "EMPTY"
MAX_RETRIES = 3
BASE_DELAY = 2
WRITING_QA_EVAL_PROMPT = """
The following is a writing-related user intent, a user query, an ideal answer, and an AI assistant's response:

[User's Intent](This is unknow to the assistant)
{intent}
[End User's Intent]

[User's Query]
{query}
[End User's Query]

[Ideal Answer]
{ground_truth}
[End Ideal Answer]

Your task is to evaluate the response from the AI assistant:

[AI assistant Response]
{response}
[End AI assistant Response]

Consider aspects such as:

- If it's a direct answer, does it address the user's intent directly?
- If it's a direct answer, does it resemble the ideal answer in structure, tone, or content?
- If it's a question, does it proactively ask clarifying questions to understand user preferences?
- If the user's intent is ambiguous or underspecified, does the assistant reasonably aim to clarify the goal or elicit more details?

Note:
- It is acceptable — and even preferable — for the assistant to ask thoughtful clarifying questions when the user intent is unclear or when multiple interpretations are possible.
- Such clarification attempts should be evaluated as helpful and aligned with intent, especially if they move the conversation toward better understanding and collaboration.

Then, give a score from 0 to 4:
- 0 = Unhelpful or off-topic
- 1 = Minimally helpful, vague or generic
- 2 = Partially helpful but missing key elements
- 3 = Helpful and mostly aligned with intent
- 4 = Very helpful, insightful, and clearly directed at achieving the intent

Please put your final score in \\boxed{{}}.
""".strip()

def extract_boxed_score(text: str) -> str:
    """
    从文本中提取 \\boxed{...} 里的内容，返回字符串。
    如果没有找到，返回 None。
    """
    match = re.search(r"\\boxed\{(\d+)\}", text)
    if match:
        number = int(match.group(1))
    else:
        print("未找到 boxed 数字")
    return number if match else 0

def call_judge(solution_str, ground_truth, judge, extra_info):
    if "chat_history" not in extra_info:
        raise ValueError(f"Missing 'chat_history' in sample: {extra_info}")
    try:
        prompt = WRITING_QA_EVAL_PROMPT.format(intent=extra_info["single_turn_prompt"], query=extra_info["chat_history"][-1]["content"], ground_truth = ground_truth, response=solution_str)
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
                max_tokens=16384,
            )
            response = output.choices[0].message.content
            print(f"judge response: {response}")
            response = extract_boxed_score(response)
            return response
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
    except:
        original_reward = {
            "score": 0,
            "answer_reward": 0,
            "format_reward": 0,
            "user_reward": 0.0,
            "llm_reward": 0.0,
            "cost_reward": 0
        }
        return original_reward
    bleu = sentence_bleu([ground_truth], solution_str)
    judge_score = call_judge(solution_str, ground_truth, judge, extra_info)
    normalized_judge_score = judge_score / 4
    normalized_judge_score = min(max(normalized_judge_score, 0.0), 1.0)
    answer_reward = max(bleu, float(normalized_judge_score))
    user_reward = 0.6 * answer_reward + 0.2 * response_reward_logistic(solution_str, L_min=200, L_max=800, k=0.005) + 0.2 * reasoning_penalty(reasoning_str,scale=200)
    llm_reward = 0.1 * format_reward1 + 0.7 * answer_reward + 0.2 * response_reward_logistic(ori_solution_str, L_min=300, L_max=2200, k=0.005)
    score = (user_reward * llm_reward) ** 0.5
    original_reward = {
        "score": score,
        "answer_reward": normalized_judge_score,
        "format_reward": format_reward1,
        "user_reward": user_reward,
        "llm_reward": llm_reward,
        "cost_reward": 0.0
    }

    print(f"format_reward1: {format_reward1}")
    print(f"answer_reward: {answer_reward}")
    return original_reward
