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
"""
Preprocess the gamellm dataset to parquet format
"""

import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs

from datasets import load_dataset, concatenate_datasets

from openai import OpenAI
import re
import json
import re
import json
from rich.console import Console
from rich.pretty import Pretty
from rich.panel import Panel
from datasets import load_dataset

import re, json, ast
from datasets import concatenate_datasets
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


INITIAL_CONTENT_PREFIX = (
    "The user's initial question is"
    "Question: {question}"
)

SYSTEM_PROMPT = """
The assistant is designed to be helpful, proactive, and highly interactive.

The assistant strives to accurately interpret the user's intent throughout the conversation, acknowledging previous interactions to maintain context and continuity. If the user's message is unclear or lacks necessary details, the assistant always asks for clarification rather than making assumptions. For example, if the user's request is incomplete, the assistant responds with: "Could you provide more details so I can assist you better?"

The assistant asks specific follow-up questions and offers suggestions based on the user's needs, avoiding vague or generic prompts. It proactively provides guidance and potential next steps, especially in complex tasks such as writing, analysis, coding, and question answering.

The assistant is mindful of how much content the user needs to read or type, keeping interactions concise and efficient. It reduces unnecessary repetition and ensures responses are relevant, well-structured, and free from errors. When presenting options or asking for feedback, the assistant simplifies interactions by offering multiple-choice answers or specific suggestions to make it easier for the user to respond quickly.

The assistant adapts its tone to align with the user's emotional state and style, adjusting its approach as needed. If uncertain about something, the assistant honestly says, "I don't know," and suggests ways for the user to find the information.

The assistant provides factually accurate, coherent, and relevant responses, using proper grammar and structure. It remains interactive and proactive across all tasks, continually seeking feedback to refine and improve interactions.
"""

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


def groupwise_split(dataset, group_key="source", test_size=0.05, seed=42):
    grouped = {}
    for example in dataset:
        key = example[group_key]
        grouped.setdefault(key, []).append(example)

    train_splits = []
    test_splits = []

    for key, examples in grouped.items():
        # 临时构建一个小 dataset
        small_dataset = dataset.select([i for i, ex in enumerate(dataset) if ex[group_key] == key])
        split = small_dataset.train_test_split(test_size=test_size, seed=seed)
        train_splits.append(split["train"])
        test_splits.append(split["test"])

    # 合并所有组的train和test
    train_dataset = concatenate_datasets(train_splits)
    test_dataset = concatenate_datasets(test_splits)

    return train_dataset, test_dataset

prompt_template = GAME_THEORY_USER_CONTENT_PREFIX

def extract_nonoverlapping_prefix(a: str, b: str) -> str:
    if not a.endswith(b):
        raise ValueError("b is not a suffix of a")
    return a[:-len(b)] if len(b) > 0 else a



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="verl/data/gamellm_wildbreak_math_medium")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    # 第一个数据集
    data_source1 = "zsqzz/abgqa-3k-formal"
    ds1 = load_dataset(data_source1)["train"]
    # 返回一个新的 Dataset（不会修改原始 ds2）
    # ds1 = ds1.remove_columns(["data_source","ability"])
    # 比如给每条样本添加一列 new_col = "default"
    ds1 = ds1.map(lambda x: {"adversarial": False})
    ds1 = ds1.map(lambda x: {"turn_id": 1})
    # 分别规范化
    # ds1 = normalize(ds1)
    # print(ds1.features["prompt"])

    # 第二个数据集
    data_source2 = "zsqzz/gamellm_wildbreak_math_medium"   # 替换成你需要的新数据集
    ds2 = load_dataset(data_source2)["train"]
    ds2 = ds2.remove_columns(["data_source", "ability","reward_model","extra_info"])

    # 合并
    full_dataset = concatenate_datasets([ds1, ds2])

    # 手动划分 90% train / 10% test
    train_dataset, test_dataset = groupwise_split(full_dataset, group_key="source", test_size=0.1)

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(row, idx):
            question_raw = row.pop("single_turn_prompt")
            answer_raw = row.pop("single_turn_completion")
            if row["source"] == "medium":
                question_raw = extract_nonoverlapping_prefix(question_raw, answer_raw)
            if row["source"] == "ambig_qa":
                assert row["prompt"][-1]["role"] == "user"
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]

                user_prompt = None
                if len(row["prompt"]) > 1:
                    user_prompt  = INITIAL_CONTENT_PREFIX.format(question=row["prompt"][0]["content"])
                    user_prompt += GAME_THEORY_USER_CONTENT_PREFIX.format(question=row["prompt"][-1]["content"])
                else:
                    user_prompt = GAME_THEORY_USER_CONTENT_PREFIX.format(question=row["prompt"][-1]["content"])
                
                messages += [{"role": "user", "content": user_prompt}]
            if row["source"] == "math-hard":
                answer_raw = extract_solution(answer_raw)
                # print(f"answer_raw: {answer_raw}")
            data = {
                "data_source": row["source"],
                "prompt": messages,
                "ability": row["source"],
                "reward_model": {"style": "rule", "ground_truth": answer_raw},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "chat_history": row["prompt"],
                    "single_turn_completion": answer_raw,
                    "single_turn_prompt": question_raw,
                    "adversarial": row["adversarial"] if row["source"] == "wildguard" else None,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
