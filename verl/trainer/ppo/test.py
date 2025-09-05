import os
file_path = "recipe/gamellm/reward_function.py"


if not os.path.exists(file_path):
    raise FileNotFoundError(f"Reward function file '{file_path}' not found.")
