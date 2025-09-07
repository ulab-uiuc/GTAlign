source /mnt/bn/heheda/miniconda3/bin/activate
conda init
conda activate game_theory_llm
tmux new

# verl node
cd /mnt/bn/heheda/verl
conda activate verl
bash /mnt/bn/heheda/verl/recipe/gamellm/run_gamellm_ppo.sh

# sgl node
cat /etc/hosts
2605:340:cd51:4900:89a6:f93f:15bd:2b51
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=0
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_IB_TIMEOUT=22
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m sglang.launch_server --host '::' --model-path /mnt/bn/heheda/HuggingFace-Download-Accelerator/hf_hub/models--Qwen--Qwen3-32B --mem-fraction-static 0.8 --port 36485 --tp 2 --dp 4