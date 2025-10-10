#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cd_vs_others_table.py

Row (fixed): cobb-douglas
Columns (opponents): user, llm, linear
Metrics (3 columns): epsilon-pareto coverage advantage, hypervolume advantage, regret advantage

Advantage definition (row better if positive):
- eps_adv  = eps_cov(CD) - eps_cov(opponent)
- hv_adv   = HV(CD) - HV(opponent)
- reg_adv  = regret(opponent) - regret(CD)   # regret is "lower is better", so flip
"""

import os
import csv
from typing import Dict
import numpy as np

# 如果你的模块名是 pareto_compare_plus，就改成那个
from pareto_ratio import load_jsonl, compute_metrics

from recipe.gamellm.tasks.format import format_reward, extract_tag_block, response_reward_logistic, reasoning_penalty, parse_payoff_block

# ---------------------- CONFIG ----------------------
# METHODS = {
#     "cobb-douglas": "verl/0910_math_cobb/125.jsonl",
# }

METHODS = {
    "wildguard": {
    # "cobb-douglas": "verl/0911_wildguard_cobb/150.jsonl",
    # "linear": "verl/0910_wildguard_linear/140.jsonl",
    "user": "verl/0911_wildguard_user/30.jsonl",
    # "llm": "verl/0910_wildguard_llm/140.jsonl",
    # "sft": "verl/0912_total_sft/0.jsonl",
    # "qwen32b": "verl/0911_total_qwen32b/0.jsonl",
    # "base": "verl/0910_total_base/0.jsonl",
    },
    "medium": {
    # "cobb-douglas": "verl/0910_medium_cobb/85.jsonl",
    # "linear": "verl/0910_medium_linear/85.jsonl",
    "user": "verl/0910_medium_user/85.jsonl",
    # "llm": "verl/0910_medium_llm/85.jsonl",
    # "sft": "verl/0912_total_sft/0.jsonl",
    # "qwen32b": "verl/0911_total_qwen32b/0.jsonl",
    # "base": "verl/0910_total_base/0.jsonl",
    },
    "math-hard": {
    # "cobb-douglas": "verl/0910_math_cobb/130.jsonl",
    # "linear": "verl/0910_math_linear/140.jsonl",
    "user": "verl/0910_math_user/110.jsonl",
    # "llm": "verl/0910_math_llm/140.jsonl",
    # "sft": "verl/0912_total_sft/0.jsonl",
    # "qwen32b": "verl/0911_total_qwen32b/0.jsonl",
    # "base": "verl/0910_total_base/0.jsonl",
    },
    "ambig_qa": {
    "cobb-douglas": "verl/0910_abgqa_cobb/60.jsonl",
    # "linear": "verl/0910_abgqa_linear/125.jsonl",
    "user": "verl/0910_abgqa_user/60.jsonl",
    # "llm": "verl/0910_abgqa_llm/125.jsonl",
    # "sft": "verl/0912_total_sft/0.jsonl",
    # "qwen32b": "verl/0911_total_qwen32b/0.jsonl",
    # "base": "verl/0910_total_base/0.jsonl",
    },
}

# METHODS = {
#     "cobb-douglas": "verl/0910_medium_cobb/85.jsonl",
# }

# METHODS = {
#     "cobb-douglas": "verl/0910_wildguard_cobb/130.jsonl",
# }


if __name__ == "__main__":


    for dataset, method in METHODS.items():
        print(f"dataset: {dataset}")    
        for key, val in method.items():
            cd_data = load_jsonl(path=val)
            format_reward_lst = []
            answer_score_lst = []
            length_lst = []
            response_length_lst = []
            length_reward_user = []
            length_reward_llm = []
            for idx, item in enumerate(cd_data):
                if item["source"] != dataset:
                    continue
                try:
                    solution_str = item["output"]
                    ori_solution_str = item["output"]
                    reasoning_str = solution_str.split("<response>")[0]
                    solution_str = extract_tag_block(solution_str,"response")
                    total_length = len(ori_solution_str)
                    response_length = len(solution_str)
                    response_length_lst.append(response_length)
                except:
                    continue
                length_lst.append(total_length)
                length_reward_user.append(0.2 * response_reward_logistic(solution_str, L_min=20, L_max=500, k=0.005) + 0.3 * reasoning_penalty(reasoning_str,scale=500))
                length_reward_llm.append(0.2 * response_reward_logistic(ori_solution_str, L_min=500, L_max=1000, k=0.005))
                format_reward_lst.append(item["format_reward"])
                if item["answer_reward"] == 0.5 and dataset == "wildguard" and item["gts"]=="harmful":
                    answer_score_lst.append(1)
                # elif dataset == "wildguard" and item["gts"]=="harmful":
                #     answer_score_lst.append(0)
                else:
                    answer_score_lst.append(item["answer_reward"])
            print(f"format_reward_lst: {np.mean(format_reward_lst)}")
            print(f"answer_score_lst: {np.mean(answer_score_lst)}")
            # print(f"reasoning_length_lst: {np.mean(length_lst)-np.mean(response_length_lst)}")
            print(f"response_length_lst: {np.mean(length_lst)}")
            # print(f"length_reward_user: {np.mean(length_reward_user)}")
            # print(f"length_reward_llm: {np.mean(length_reward_llm)}")
            print(f"Answer Score per Total Token: {1000 * np.mean(answer_score_lst) / np.mean(length_lst)}")
            # print(f"Answer Score per Reasoning Token: {1000 * np.mean(answer_score_lst) / (np.mean(length_lst) - np.mean(response_length_lst))}")
            # print(f"Answer Score per Response Token: {1000 * np.mean(answer_score_lst) / np.mean(response_length_lst)}")
            print(f"Reasoning-Response Ratio (RRR): {np.mean(response_length_lst) / np.mean(length_lst)}")
            # print(f"quality_cost_efficiency: {np.mean(format_reward_lst)}")
            print(f"key: {key}\n")
        print("\n\n\n")
        
