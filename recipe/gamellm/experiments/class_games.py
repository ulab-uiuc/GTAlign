import pandas as pd
import matplotlib.pyplot as plt

# File paths and reward columns mapping
files = [
    "verl/recipe/gamellm/experiments/class_games/Signaling-generalization-2025-9-11_13_27_49.csv",
    "verl/recipe/gamellm/experiments/class_games/Signaling-generalization-2025-9-11_13_27_57.csv",
    "verl/recipe/gamellm/experiments/class_games/Signaling-generalization-2025-9-11_13_28_31.csv",
    "verl/recipe/gamellm/experiments/class_games/Signaling-generalization-2025-9-11_13_28_43.csv"
]

reward_columns = {
    "File 1": "Signaling-generalization-train/PrisonersDilemma/player2_round_reward_step",
    "File 2": "Signaling-generalization-train/Rubinstein/reward_step",
    "File 3": "Signaling-generalization-train/Signaling/reward_step",
    "File 4": "Signaling-generalization-train/Signalingp2/reward_step"
}

titles = [
    "Sequential Prisoner's Dilemma",
    "Rubinstein Bargaining Game",
    "Sequential Signaling Game"
]

selected_files = [files[0], files[1], files[3]]
selected_reward_cols = [
    reward_columns["File 1"],
    reward_columns["File 2"],
    reward_columns["File 4"]
]

# --------- 单独子图，不统一 y 轴 ---------
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)

for ax, file, title, reward_col in zip(axes, selected_files, titles, selected_reward_cols):
    df = pd.read_csv(file)
    ax.plot(df['step'], df[reward_col], marker='o')
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")

    # 让每个折线在自己的坐标轴范围内居中
    ymin, ymax = df[reward_col].min(), df[reward_col].max()
    margin = (ymax - ymin) * 0.2  # 上下加 20% 边距
    ax.set_ylim(ymin - margin, ymax + margin)

# plt.suptitle("Step vs Reward (Three Selected Games, Individually Centered Y-axis)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
plt.savefig("verl/recipe/gamellm/experiments/step_vs_reward_three_games_centered.png")

# --------- Overlay 图，保持各自居中 ---------
plt.figure(figsize=(8, 5))
colors = ["tab:blue", "tab:orange", "tab:green"]
labels = [
    "Sequential Prisoner's Dilemma",
    "Rubinstein Game",
    "Sequential Signaling Game"
]

for file, reward_col, color, label in zip(selected_files, selected_reward_cols, colors, labels):
    df = pd.read_csv(file)
    plt.plot(df['step'], df[reward_col], marker='o', label=label, color=color)

plt.xlabel("Step")
plt.ylabel("Reward")
# plt.title("Step vs Reward (Overlayed Curves, Each Centered Locally)")
plt.legend()
plt.grid(False)
plt.show()
plt.savefig("verl/recipe/gamellm/experiments/step_vs_reward_three_games_overlay_centered.png")