# vllm server
# CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve verl-team/GenRM-CI-Test-1.5B --served_model_name genrm-demo

# sglang server
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang_router.launch_server --model-path verl-team/GenRM-CI-Test-1.5B --dp-size 4

# export CUDA_DEVICE_MAX_CONNECTIONS=1
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_IB_GID_INDEX=0
# export NCCL_IB_DISABLE=1
# export NCCL_NET_GDR_LEVEL=2
# export NCCL_IB_QPS_PER_CONNECTION=4
# export NCCL_IB_TC=160
# export NCCL_IB_TIMEOUT=22
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m sglang.launch_server --host '::' --model-path /mnt/bn/heheda/HuggingFace-Download-Accelerator/hf_hub/models--Qwen--Qwen3-32B --mem-fraction-static 0.8 --port 30003 --tp 2 --dp 4
# CUDA_VISIBLE_DEVICES=0,1 python3 -m sglang.launch_server --model-path /mnt/bn/heheda/HuggingFace-Download-Accelerator/hf_hub/models--Qwen--Qwen3-32B --mem-fraction-static 0.8 --port 30003 --tp 2 --dp 1

# # Test on verl node
# curl -v "http://[2605:340:cd51:4900:514d:6685:7e57:86c4]:30003/v1/models"

# # Call from verl node (HTTP/URL with brackets)
# curl -X POST "http://[2605:340:cd51:4900:6e28:2bce:5eb:2f38]:36485/v1/chat/completions" \
#   -H "Content-Type: application/json" \
#   -d '{"model":"default","messages":[{"role":"user","content":"hello"}]}'


set -x

HOME=/mnt/bn/heheda/verl

# For async rollout mode, dataset should return raw chat.
rollout_mode="sync"
if [ "$rollout_mode" = "async" ]; then
    return_raw_chat="True"
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files=/mnt/bn/heheda/verl/data/gamellm_wildbreak_math_medium_abgqa/by_source/medium/train.parquet \
    data.val_files=/mnt/bn/heheda/verl/data/gamellm_wildbreak_math_medium_abgqa/by_source/medium/test.parquet \
    data.return_raw_chat=$return_raw_chat \
    data.train_batch_size=512 \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/mnt/bn/heheda/HuggingFace-Download-Accelerator/hf_hub/models--Qwen--Qwen2.5-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 \
    actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
    critic.optim.lr=2e-6 \
    critic.model.use_remove_padding=True \
    critic.model.path=/mnt/bn/heheda/HuggingFace-Download-Accelerator/hf_hub/models--Qwen--Qwen2.5-3B-Instruct \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_max_token_len_per_gpu=98304 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=gamellm \
    +reward_model.judge_ip=localhost \
    +reward_model.judge_port=36485 \
    custom_reward_function.path=recipe/gamellm/reward_function_abgqa_v2.py \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='siqi_verl_gamellm' \
    trainer.experiment_name='0910_medium_cobb' \
    trainer.n_gpus_per_node=4 \
    trainer.validation_data_dir='/mnt/bn/heheda/verl/0910_medium_cobb' \
    trainer.val_before_train=True \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=30 $@
