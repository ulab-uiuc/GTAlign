# GTAlign: Game-Theoretic Alignment of LLM Assistants for Social Welfare

[![ArXiv](https://img.shields.io/badge/arXiv-2510.08872-b31b1b?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2510.08872)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-GTAlign-orange?logo=huggingface&logoColor=white)](https://huggingface.co/GTAlign)
<!-- [![License](https://img.shields.io/badge/LICENSE-Apache2.0-green)](https://github.com/ulab-uiuc/GTAlign/blob/main/LICENSE) -->



**GTAlign** applies game-theoretic principles to fine-tune reasoning LLMs, encouraging them to make decisions that are not only accurate but also rational, cooperative, and transparent in dialogue settings.


## Getting Started


### Installation

```shell
# create a new environment
conda create -n gtalign python==3.10
conda activate gtalign

# Install dependencies
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install -r requirements.txt

# Install GTAlign
git clone https://github.com/ulab-uiuc/GTAlign.git
cd GTAlign
pip install --no-deps -e .
```

### Experiments

We recommend using eight NVIDIA H20 GPUs for our experiments.

#### Launch LLM judge using sglang:
```bash
## get ipv6/ipv4 address
cat /etc/hosts

## run judge model on a node different from verl node, using ipv6.
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=0
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_IB_TIMEOUT=22
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m sglang.launch_server --model-path Qwen/Qwen3-32B --host '::' --mem-fraction-static 0.8 --port port_number --tp 1 --dp 8

# Test on verl node
curl -v "http://[ipv6]:port_number/v1/models"

# Call from verl node (HTTP/URL with brackets)
curl -X POST "http://[ipv6]:port_number/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"hello"}]}'

## serve judge model on the same node as the verl node.
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m sglang.launch_server --model-path Qwen/Qwen3-32B --mem-fraction-static 0.8 --port port_number --tp 4 --dp 1 --reasoning-parser qwen3
```

#### Run RL training:
you need to change the model and data path in the scripts.
```bash
# verl node
bash recipe/gamellm/math.sh
bash recipe/gamellm/medium.sh
bash recipe/gamellm/adgqa.sh
bash recipe/gamellm/wildguard.sh
bash recipe/gamellm/full_zscore.sh
```

#### Run OOD evaluation:
you need to change the model and data path in the scripts.
```bash
# transform fsdp checkpoint to huggingface checkpoint
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir xxx \
    --target_dir xxx

# run evaluation scripts
bash recipe/gamellm/ablation_advbench.sh
bash recipe/gamellm/ablation_coqa.sh
bash recipe/gamellm/ablation_minerva.sh
```

### Dataset
The dataset used in RL training and evaluation is at `data/`

### Inference
We provide inference scripts at `recipe/gamellm/inference/inference_v2.py`. You could modify the payoff matrix during inference to control model behavior.

## References

This project builds upon the following open-source repositories:

- [verl](https://github.com/volcengine/verl) License: [Apache License 2.0](https://github.com/volcengine/verl/blob/main/LICENSE)

- [collabllm](https://github.com/Wuyxin/collabllm) License: [MIT License](https://github.com/Wuyxin/collabllm/blob/main/LICENSE)


## Citation

If you find our work helpful to your research, please cite as follows:
```bibtex
@misc{zhu2025gtaligngametheoreticalignmentllm,
      title={GTAlign: Game-Theoretic Alignment of LLM Assistants for Social Welfare}, 
      author={Siqi Zhu and David Zhang and Pedro Cisneros-Velarde and Jiaxuan You},
      year={2025},
      eprint={2510.08872},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.08872}, 
}
```
