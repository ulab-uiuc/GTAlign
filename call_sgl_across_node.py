# launch on sglang node
'''
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=0
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_IB_TIMEOUT=22
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m sglang.launch_server --host '::' --model-path /mnt/bn/heheda/HuggingFace-Download-Accelerator/hf_hub/models--Qwen--Qwen3-4B --mem-fraction-static 0.8 --port 36485 --tp 1 --dp 8
CUDA_VISIBLE_DEVICES=0,1 python3 -m sglang.launch_server --model-path /mnt/bn/heheda/HuggingFace-Download-Accelerator/hf_hub/models--Qwen--Qwen3-32B --mem-fraction-static 0.8 --port 30003 --tp 2 --dp 1
'''

# Test on verl node
'''curl -v "http://[2605:340:cd51:4900:6e28:2bce:5eb:2f38]:36485/v1/models"'''

# Call from verl node (HTTP/URL with brackets)
'''
curl -X POST "http://[2605:340:cd51:4900:6e28:2bce:5eb:2f38]:36485/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"hello"}]}'
'''

from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:30003/v1"
)

resp = client.chat.completions.create(
    model="default",
    messages=[{"role":"user","content":"Hello over IPv6!"}]
)
print(resp.choices[0].message.content)