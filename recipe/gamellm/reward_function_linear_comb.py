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

def reward_func(data_source, solution_str, ground_truth, judge: None, extra_info=None):
    if data_source in ["medium"]:
        from recipe.gamellm.tasks_linear_comb import medium
        return medium.compute_score(solution_str, ground_truth, judge, extra_info)
    elif data_source == "math-hard":
        from recipe.gamellm.tasks_linear_comb import math_hard
        return math_hard.compute_score(solution_str, ground_truth, judge, extra_info)
    elif data_source in ["wildguard"]:
        from recipe.gamellm.tasks_linear_comb import wildguard
        return wildguard.compute_score(solution_str, ground_truth, judge, extra_info)
    elif data_source in ["ambig_qa"]:
        from recipe.gamellm.tasks_linear_comb import abgqa_v2
        return abgqa_v2.compute_score(solution_str, ground_truth, judge, extra_info)
    else:
        raise NotImplementedError

def compute_score(data_source, solution_str, ground_truth, judge, extra_info):
    reward_score = reward_func(data_source, solution_str, ground_truth, judge, extra_info)
    reward_score["source"] = data_source
    return reward_score