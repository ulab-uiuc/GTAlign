# verl node
cd verl
conda activate verl
bash verl/recipe/gamellm/xxx.sh

# sgl node

## get ipv6/ipv4 address
cat /etc/hosts

## serve judge model on a different node, using ipv6.
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=0
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_IB_TIMEOUT=22
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m sglang.launch_server --model-path xxx --host '::' --mem-fraction-static 0.8 --port 36485 --tp 1 --dp 8

## serve judge model on the same node as the verl training.
CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server --model-path xxx --mem-fraction-static 0.8 --port 36486 --tp 1 --dp 1
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m sglang.launch_server --model-path xxx --mem-fraction-static 0.8 --port 36485 --tp 4 --dp 1 --reasoning-parser qwen3

## transform fsdp checkpoint to huggingface checkpoint
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir xxx \
    --target_dir xxx
