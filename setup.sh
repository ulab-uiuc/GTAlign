source /mnt/bn/heheda/miniconda3/bin/activate
conda init
conda activate game_theory_llm
tmux new

# verl node
cd /mnt/bn/heheda/verl
conda activate verl
bash /mnt/bn/heheda/verl/recipe/gamellm/run_gamellm_ppo.sh

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m sglang.launch_server --model-path /mnt/bn/heheda/HuggingFace-Download-Accelerator/hf_hub/models--Qwen--Qwen3-32B --mem-fraction-static 0.8 --port 36485 --tp 4 --dp 1 --reasoning-parser qwen3

# sgl node
cat /etc/hosts
2605:340:cd51:4900:7f7b:998f:86d0:a3b0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=0
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_IB_TIMEOUT=22
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m sglang.launch_server --model-path /mnt/bn/heheda/HuggingFace-Download-Accelerator/hf_hub/models--Qwen--Qwen3-32B --host '::' --mem-fraction-static 0.8 --port 36485 --tp 1 --dp 8

CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server --model-path /mnt/bn/heheda/verl/checkpoints/siqi_verl_gamellm/0910_abgqa_cobb/global_step_140/actor_merged --mem-fraction-static 0.8 --port 36486 --tp 1 --dp 1
CUDA_VISIBLE_DEVICES=1 python3 -m sglang.launch_server --model-path /mnt/bn/heheda/verl/checkpoints/siqi_verl_gamellm/0911_wildguard_cobb/global_step_140/actor_merged --mem-fraction-static 0.8 --port 36487 --tp 1 --dp 1



python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /mnt/bn/heheda/verl/checkpoints/siqi_verl_gamellm/0910_abgqa_cobb/global_step_140/actor \
    --target_dir /mnt/bn/heheda/verl/checkpoints/siqi_verl_gamellm/0910_abgqa_cobb/global_step_140/actor_merged

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /mnt/bn/heheda/verl/checkpoints/siqi_verl_gamellm/0911_wildguard_cobb/global_step_140/actor \
    --target_dir /mnt/bn/heheda/verl/checkpoints/siqi_verl_gamellm/0911_wildguard_cobb/global_step_140/actor_merged


python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /mnt/bn/heheda/verl/checkpoints/siqi_verl_gamellm/0910_math_cobb/global_step_140/actor \
    --target_dir /mnt/bn/heheda/verl/checkpoints/siqi_verl_gamellm/0910_math_cobb/global_step_140/actor_merged


