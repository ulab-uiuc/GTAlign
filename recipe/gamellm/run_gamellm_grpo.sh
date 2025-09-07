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


# 需要在verl根目录运行
set -x

# export SWANLAB_API_KEY=lEwCGfEJsM2C0hqvxlcuV
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/bn/heheda/verl/data/gamellm_wildbreak_math_medium_abgqa/train.parquet \
    data.val_files=/mnt/bn/heheda/verl/data/gamellm_wildbreak_math_medium_abgqa/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/mnt/bn/heheda/gamellm_new/outputs/sft/gamellm/thinking-full-parallel-test3/checkpoint-18 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=gamellm \
    +reward_model.judge_ip=2605:340:cd51:4900:89a6:f93f:15bd:2b51 \
    +reward_model.judge_port=36485 \
    custom_reward_function.path=recipe/gamellm/reward_function.py \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='siqi_verl_gamellm' \
    trainer.experiment_name='gamellm_grpo_0906_test1' \
    trainer.n_gpus_per_node=8 \
    trainer.validation_data_dir='/mnt/bn/heheda/verl/exp0906_grpo' \
    trainer.val_before_train=True \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=20 \
    trainer.total_epochs=10 \
    trainer.resume_mode='disable' $@
