# Copyright 2024 GameLLM team and/or its affiliates
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

import asyncio

from concurrent.futures import ThreadPoolExecutor  # Changed from ProcessPoolExecutor to avoid pickle issues
from functools import partial
from typing import Callable, Optional

import psutil
import torch
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from openai import OpenAI
from collections import defaultdict
import numpy as np

async def single_compute_score(evaluation_func, completion, reference, task, judge, task_extra_info, executor, timeout=9000.0):
    loop = asyncio.get_running_loop()
    try:
        # Ensure process_completion is called properly
        future = loop.run_in_executor(executor, partial(evaluation_func, task, completion, reference, judge, task_extra_info))
        return await asyncio.wait_for(future, timeout=timeout)
    except asyncio.TimeoutError:
        print(f"[Timeout] Task timeout: {completion}")
        return None  # Default value for timed-out rows
    except Exception as e:
        print(f"[Error] Task failed: {e}, completion: {completion[:80]}")
        return None  # Default value for failed rows


async def parallel_compute_score_async(
    evaluation_func, completions, references, tasks, extra_info=None, judge_client=None, num_processes=64
):
    if extra_info is None:
        extra_info = [None] * len(tasks)
    scores = []
    with ThreadPoolExecutor(max_workers=num_processes) as executor:  # Changed from ProcessPoolExecutor
        # ThreadPoolExecutor shares memory with main process, avoiding pickle serialization issues
        # Note: For CPU-bound tasks, ProcessPoolExecutor is better, but for I/O-bound HTTP calls, ThreadPoolExecutor works well
        try:
            # Create tasks for all rows
            tasks_async = [
                single_compute_score(evaluation_func, c, r, t, judge_client, ei, executor,timeout=300.0)
                for c, r, t, ei in zip(completions, references, tasks, extra_info, strict=True)
            ]
            results = await asyncio.gather(*tasks_async, return_exceptions=False)
        except Exception as e:
            print(f"[Exception] async gather failed: {e}")
            raise
        finally:
            # Only cleanup processes for ProcessPoolExecutor
            if hasattr(executor, '_processes'):
                terminated_count = 0
                for pid, proc in executor._processes.items():
                    try:
                        p = psutil.Process(pid)
                        p.terminate()
                        try:
                            p.wait(timeout=5)
                        except psutil.TimeoutExpired:
                            p.kill()
                        terminated_count += 1
                    except Exception:
                        pass
                print(f"[Shutdown] {terminated_count} subprocess(es) terminated.")
            # ThreadPoolExecutor threads will be cleaned up automatically

    # Process results
    for result, completion, reference, task in zip(results, completions, references, tasks, strict=True):
        if isinstance(result, Exception) or result is None:
            # Handle failed or timed-out tasks
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
                "source": task
            }
            scores.append(original_reward)
            # print("line86 exception reward")
        elif isinstance(result, int | float | bool):
            scores.append(float(result))
            # print("line89 float reward")
        elif isinstance(result, dict):
            # print("line89 dict reward")
            scores.append(result)
        else:
            # print("line94 others reward")
            scores.append(float(result[0]))
    return scores


def run_reward_scoring(evaluation_func, completions, references, tasks, extra_info=None, judge_client=None, num_processes=64):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            parallel_compute_score_async(evaluation_func, completions, references, tasks, extra_info, judge_client, num_processes)
        )
    finally:
        loop.close()


from collections import defaultdict

def listdict_to_dictlist(list_of_dict):
    out = defaultdict(list)
    for d in list_of_dict:
        for k, v in d.items():
            out[k].append(v)
    return dict(out)


@register("gamellm")
class GameLLMRewardManager:
    """
    The Reward Manager used in gamellm
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int,
        compute_score: Optional[Callable] = None,
        reward_fn_key: str = "data_source",
        judge_ip: str = "localhost",
        judge_port: str = "30003",
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.judge_ip = judge_ip
        self.judge_port = judge_port
        if self.judge_ip == "localhost":
            self.judge_url = f"http://{self.judge_ip}:{self.judge_port}/v1" # ipv6的ip需要用中括号括起来
        else:
            self.judge_url = f"http://[{self.judge_ip}]:{self.judge_port}/v1" # ipv6的ip需要用中括号括起来
        self.judge_client = OpenAI(
            api_key="EMPTY",
            base_url=self.judge_url,
        )
        
    def verify(self, data):
        """
        verify the batch and save as ``acc`` tensor
        """
        # batched scoring
        prompt_ids = data.batch["prompts"]

        response_ids = data.batch["responses"]
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        ground_truth = [data_item.non_tensor_batch["reward_model"]["ground_truth"] for data_item in data]
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        extra_info = data.non_tensor_batch.get("extra_info", None)

        assert len(sequences_str) == len(ground_truth) == len(data_sources)
        try:
            scores = run_reward_scoring(
                self.compute_score,
                completions=sequences_str,
                references=ground_truth,
                tasks=data_sources,
                extra_info=extra_info,
                num_processes=1024,
                judge_client=self.judge_client,
            )
        except asyncio.TimeoutError:
            print("[Timeout] Global reward scoring timed out. Setting all as 0.")
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
            
            scores = [original_reward for _ in range(len(sequences_str))]
            for idx, item in enumerate(scores):
                item["source"] = data_sources[idx]
        except Exception as e:
            print(f"[Error] Unexpected error during scoring. Setting all as 0. {e}")
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
            scores = [original_reward for _ in range(len(sequences_str))]
            for idx, item in enumerate(scores):
                item["source"] = data_sources[idx]
        scores = listdict_to_dictlist(scores)
        data.batch["acc"] = torch.tensor(scores["score"], dtype=torch.float32, device=prompt_ids.device)
        
        return scores

    def __call__(self, data: DataProto, return_dict: bool = True):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        already_print_data_sources = {}

        # batched scoring
        prompt_ids = data.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]

        response_ids = data.batch["responses"]
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1)
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        data_sources = data.non_tensor_batch["data_source"]

        scores = self.verify(data)
        # print("DEBUG scores:", scores[:5])

        for i in range(len(data)):
            data_source = data_sources[i]
            reward_tensor[i, valid_response_length[i].item() - 1] = scores["score"][i]

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                # print(sequences_str)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": scores}
        else:
            return reward_tensor



from collections import defaultdict
import numpy as np

def robust_z(item_scores_by_src, clip_q=(0.01, 0.99), k=5.0):
    stats = {}
    for src, vals in item_scores_by_src.items():
        vals = np.asarray(vals, dtype=np.float32)
        # winsorize
        lo, hi = np.quantile(vals, clip_q[0]), np.quantile(vals, clip_q[1])
        vals_w = np.clip(vals, lo, hi)
        med = float(np.median(vals_w))
        mad = float(np.median(np.abs(vals_w - med))) * 1.4826
        if mad < 1e-6: mad = 1.0
        stats[src] = (med, mad, lo, hi)
    return stats




@register("gamellm_normalize")
class GameLLMRewardNormalizeManager:
    """
    The Reward Manager used in gamellm
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int,
        compute_score: Optional[Callable] = None,
        reward_fn_key: str = "data_source",
        judge_ip: str = "localhost",
        judge_port: str = "30003",
        normalize_method: str = "minmax",
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.judge_ip = judge_ip
        self.judge_port = judge_port
        self.normalize_method = normalize_method
        if self.judge_ip == "localhost":
            self.judge_url = f"http://{self.judge_ip}:{self.judge_port}/v1" # ipv6的ip需要用中括号括起来
        else:
            self.judge_url = f"http://[{self.judge_ip}]:{self.judge_port}/v1" # ipv6的ip需要用中括号括起来
        self.judge_client = OpenAI(
            api_key="EMPTY",
            base_url=self.judge_url,
        )
        
    def verify(self, data):
        """
        verify the batch and save as ``acc`` tensor
        """
        # batched scoring
        prompt_ids = data.batch["prompts"]

        response_ids = data.batch["responses"]
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        ground_truth = [data_item.non_tensor_batch["reward_model"]["ground_truth"] for data_item in data]
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        extra_info = data.non_tensor_batch.get("extra_info", None)

        assert len(sequences_str) == len(ground_truth) == len(data_sources)
        try:
            scores = run_reward_scoring(
                self.compute_score,
                completions=sequences_str,
                references=ground_truth,
                tasks=data_sources,
                extra_info=extra_info,
                num_processes=1024,
                judge_client=self.judge_client,
            )
        except asyncio.TimeoutError:
            print("[Timeout] Global reward scoring timed out. Setting all as 0.")
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
            
            scores = [original_reward for _ in range(len(sequences_str))]
            for idx, item in enumerate(scores):
                item["source"] = data_sources[idx]
        except Exception as e:
            print(f"[Error] Unexpected error during scoring. Setting all as 0. {e}")
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
            scores = [original_reward for _ in range(len(sequences_str))]
            for idx, item in enumerate(scores):
                item["source"] = data_sources[idx]
        

        ## normalize the scores
        normalize_mode = self.normalize_method  # 可选: "zscore", "minmax", "none"

        if normalize_mode != "none":
            # 收集每个数据集的 score
            grouped_scores = defaultdict(list)
            for item in scores:
                grouped_scores[item["source"]].append(item["score"])

            stats = {}
            if normalize_mode == "zscore":
                for source, vals in grouped_scores.items():
                    vals = np.array(vals, dtype=np.float32)
                    mean = float(vals.mean())
                    std = float(vals.std()) if vals.std() > 1e-6 else 1.0
                    stats[source] = (mean, std)

                epsilon = 1e-6
                for item in scores:
                    mean, std = stats[item["source"]]
                    item["ppo_score"] = (item["score"] - mean) / (std + epsilon)

            elif normalize_mode == "minmax":
                for source, vals in grouped_scores.items():
                    vals = np.array(vals, dtype=np.float32)
                    vmin, vmax = float(vals.min()), float(vals.max())
                    if abs(vmax - vmin) < 1e-6:
                        stats[source] = (vmin, vmax, True)
                    else:
                        stats[source] = (vmin, vmax, False)
                for item in scores:
                    vmin, vmax, is_const = stats[item["source"]]
                    if is_const:
                        item["ppo_score"] = 0.0
                    else:
                        item["ppo_score"] = (item["score"] - vmin) / (vmax - vmin)
            elif normalize_mode == "robustz":
                import numpy as np
                stats = robust_z(grouped_scores)
                for it in scores:
                    med, mad, lo, hi = stats[it["source"]]
                    x = np.clip(it["score"], lo, hi)
                    z = (x - med) / mad
                    it["ppo_score"] = float(np.tanh(z / 2.0))
            elif normalize_mode == "cdf":
                import numpy as np
                import scipy.stats as st

                rank_maps = {}
                for src, vals in grouped_scores.items():
                    vals = np.asarray(vals, dtype=np.float32)
                    ranks = st.rankdata(vals, method="average")
                    p = (ranks - 0.5) / len(vals)
                    order = np.argsort(vals)
                    rank_maps[src] = (vals[order], p[order])

                def value_to_p(src, x):
                    xs, ps = rank_maps[src]
                    return float(np.interp(x, xs, ps, left=ps[0], right=ps[-1]))

                for it in scores:
                    p = value_to_p(it["source"], it["score"])
                    it["ppo_score"] = 2.0 * p - 1.0
            elif normalize_mode == "quantile":
                import numpy as np
                all_vals = np.asarray([it["score"] for it in scores], dtype=np.float32)
                all_vals.sort()
                def F_global_inv(p):
                    idx = int(np.clip(p, 1e-6, 1-1e-6) * (len(all_vals)-1))
                    return float(all_vals[idx])

                cdf_maps = {}
                for src, vals in grouped_scores.items():
                    xs = np.sort(np.asarray(vals, dtype=np.float32))
                    ps = (np.arange(len(xs)) + 0.5) / len(xs)
                    cdf_maps[src] = (xs, ps)

                def Fg(x, src):
                    xs, ps = cdf_maps[src]
                    return float(np.interp(x, xs, ps, left=ps[0], right=ps[-1]))

                for it in scores:
                    p = Fg(it["score"], it["source"])
                    it["ppo_score"] = F_global_inv(p)  # 也可以改成 st.norm.ppf(p)
            elif normalize_mode == "james":
                import numpy as np
                all_vals = np.asarray([it["score"] for it in scores], dtype=np.float32)
                mu_glob, std_glob = float(all_vals.mean()), float(all_vals.std() + 1e-6)

                kappa = 50  # 超参，可根据 batch 总量调
                stats = {}
                for src, vals in grouped_scores.items():
                    vals = np.asarray(vals, dtype=np.float32)
                    n = len(vals)
                    mu, sd = float(vals.mean()), float(vals.std() + 1e-6)
                    lam = n / (n + kappa)
                    mu_s = lam * mu + (1 - lam) * mu_glob
                    sd_s = np.sqrt(lam * (sd**2) + (1 - lam) * (std_glob**2))
                    stats[src] = (mu_s, sd_s)

                for it in scores:
                    mu_s, sd_s = stats[it["source"]]
                    it["ppo_score"] = (it["score"] - mu_s) / (sd_s + 1e-6)
            elif normalize_mode == "decentralize":
                import numpy as np
                mu_by_src = {s: float(np.mean(v)) for s, v in grouped_scores.items()}
                all_vals = np.asarray([it["score"] for it in scores], dtype=np.float32)
                std_glob = float(all_vals.std() + 1e-6)

                for it in scores:
                    it["ppo_score"] = (it["score"] - mu_by_src[it["source"]]) / std_glob
            elif normalize_mode == "selfadapt":
                import numpy as np
                tau = 2.0
                mu_by_src = {s: float(np.mean(v)) for s, v in grouped_scores.items()}
                for it in scores:
                    it["ppo_score"] = float(np.tanh((it["score"] - mu_by_src[it["source"]]) / tau))

        scores = listdict_to_dictlist(scores)
        data.batch["acc"] = torch.tensor(scores["ppo_score"], dtype=torch.float32, device=prompt_ids.device)
        
        return scores

    def __call__(self, data: DataProto, return_dict: bool = True):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        already_print_data_sources = {}

        # batched scoring
        prompt_ids = data.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]

        response_ids = data.batch["responses"]
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1)
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        data_sources = data.non_tensor_batch["data_source"]

        scores = self.verify(data)
        # print("DEBUG scores:", scores[:5])

        for i in range(len(data)):
            data_source = data_sources[i]
            reward_tensor[i, valid_response_length[i].item() - 1] = scores["ppo_score"][i]

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                # print(sequences_str)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": scores}
        else:
            return reward_tensor
