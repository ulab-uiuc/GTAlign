from huggingface_hub import upload_folder

# upload_folder(
#     folder_path="verl/checkpoints/siqi_verl_gamellm/0911_wildguard_cobb/global_step_140/actor_merged",       # 你的本地模型目录
#     repo_id="GTAlign/Qwen2.5-3B-WildGuard-140step",          # 仓库ID（例如 "siqi-zhu/my-model"）
#     repo_type="model",                        # 模型类型
#     token="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"
# )

# upload_folder(
#     folder_path="verl/checkpoints/siqi_verl_gamellm/0910_abgqa_cobb/global_step_140/actor_merged",       # 你的本地模型目录
#     repo_id="GTAlign/Qwen2.5-3B-AbgQA-140step",          # 仓库ID（例如 "siqi-zhu/my-model"）
#     repo_type="model",                        # 模型类型
#     token="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"
# )

# upload_folder(
#     folder_path="verl/checkpoints/siqi_verl_gamellm/0910_math_cobb/global_step_140/actor_merged",       # 你的本地模型目录
#     repo_id="GTAlign/Qwen2.5-3B-Math-140step",          # 仓库ID（例如 "siqi-zhu/my-model"）
#     repo_type="model",                        # 模型类型
#     token="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"
# )

# upload_folder(
#     folder_path="verl/checkpoints/siqi_verl_gamellm/0910_medium_cobb/global_step_110/actor_merged",       # 你的本地模型目录
#     repo_id="GTAlign/Qwen2.5-3B-Medium-110step",          # 仓库ID（例如 "siqi-zhu/my-model"）
#     repo_type="model",                        # 模型类型
#     token="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"
# )


upload_folder(
    folder_path="verl/checkpoints/siqi_verl_gamellm/1006_alldataset_normalize_zscore/global_step_160/actor_merged",       # 你的本地模型目录
    repo_id="GTAlign/Qwen2.5-3B-Full-160step",          # 仓库ID（例如 "siqi-zhu/my-model"）
    repo_type="model",                        # 模型类型
    token="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"
)

