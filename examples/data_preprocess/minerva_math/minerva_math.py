import pandas as pd
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi

# --- 1. 配置 ---
# !!! 重要：请将下面的字符串替换为您自己的 Hugging Face 用户名 !!!
HF_USERNAME = "zsqzz"

# 您想创建的新数据集仓库的名称
NEW_DATASET_NAME = "minervamath"
SOURCE_DATASET = "math-ai/minervamath"
NEW_DATASET_REPO_ID = f"{HF_USERNAME}/{NEW_DATASET_NAME}"
LOCAL_PARQUET_PATH = "/mnt/bn/heheda/verl/data/minerva_math/test.parquet"


# --- 2. 下载源数据集 ---
print(f"正在从 '{SOURCE_DATASET}' 下载数据集...")
# 为了快速验证，您可以先加载一小部分数据
# production 时可以移除 take(100)
source_ds = Dataset.from_list(list(load_dataset(SOURCE_DATASET, split='test')))
print(f"数据集下载完成。已加载 {len(source_ds)} 条记录。")


# --- 3. 定义转换逻辑 ---
# 从图2中提取的固定 prompt
SYSTEM_PROMPT = """
The assistant is designed to be helpful, proactive, and highly interactive.

The assistant strives to accurately interpret the user's intent throughout the conversation, acknowledging previous interactions to maintain context and continuity. If the user's message is unclear or lacks necessary details, the assistant always asks for clarification rather than making assumptions. For example, if the user's request is incomplete, the assistant responds with: "Could you provide more details so I can assist you better?"

The assistant asks specific follow-up questions and offers suggestions based on the user's needs, avoiding vague or generic prompts. It proactively provides guidance and potential next steps, especially in complex tasks such as writing, analysis, coding, and question answering.

The assistant is mindful of how much content the user needs to read or type, keeping interactions concise and efficient. It reduces unnecessary repetition and ensures responses are relevant, well-structured, and free from errors. When presenting options or asking for feedback, the assistant simplifies interactions by offering multiple-choice answers or specific suggestions to make it easier for the user to respond quickly.

The assistant adapts its tone to align with the user's emotional state and style, adjusting its approach as needed. If uncertain about something, the assistant honestly says, "I don't know," and suggests ways for the user to find the information.

The assistant provides factually accurate, coherent, and relevant responses, using proper grammar and structure. It remains interactive and proactive across all tasks, continually seeking feedback to refine and improve interactions.
"""

GAME_THEORY_USER_CONTENT_PREFIX = (
    "Now you need to answer the user's most recent question. Always produce output in EXACTLY four tagged blocks, in this order: "
    "<thinking>...</thinking><payoff>...</payoff><analyze>...</analyze><response>...</response>. "
    "Do not include anything outside these tags. Do not use markdown, code fences, or extra commentary."

    "Inside <thinking>...</thinking>: perform a concise game-theoretic analysis of strategy choices for this turn. "
    "Consider three assistant strategies: DA = direct answer; CQ = clarifying question only; "
    "AQ = answer + ONE targeted follow-up question. "
    "Also classify the user's question type as either DQ = definite/clear or VQ = vague/ambiguous. "
    "Discuss how each assistant strategy would impact both parties (LLM vs user) in terms of: "
    "answer correctness risk, ambiguity resolution, user time/effort, token/compute cost, satisfaction, and likelihood of success. "
    "Keep this analysis brief but concrete (2-5 sentences). No numbers here yet."

    "Inside <payoff>...</payoff>: output a STRICT JSON object (no trailing commas, no comments) with EXACTLY these six keys: "
    "{{\"DQ_AQ\", \"DQ_CQ\", \"DQ_DA\", \"VQ_AQ\", \"VQ_CQ\", \"VQ_DA\"}}. "
    "Each key maps to an object with two float fields: {{\"LLM\": <float>, \"user\": <float>}}. "
    "Use a consistent scale where higher is better; 0 = neutral; negatives allowed for costs; keep values roughly in [-5.0, 5.0]. "
    "Numbers must have at most one decimal place and reflect your <thinking> analysis. Example shape (values are examples only): "
    "{{ \"DQ_AQ\": {{ \"LLM\": 2.2, \"user\": 1.9 }}, \"DQ_CQ\": {{ \"LLM\": 3.1, \"user\": 3.5 }}, "
    "\"DQ_DA\": {{ \"LLM\": 4.2, \"user\": 4.3 }}, \"VQ_AQ\": {{ \"LLM\": 2.0, \"user\": 2.1 }}, "
    "\"VQ_CQ\": {{ \"LLM\": 1.3, \"user\": 1.2 }}, \"VQ_DA\": {{ \"LLM\": 2.0, \"user\": 1.8 }} }}."

    "Inside <analyze>...</analyze>: derive the Pareto frontier from the six payoff points (treat each as (LLM, user)). "
    "List the non-dominated pairs and briefly justify why dominated ones are excluded. "
    "Then pick ONE recommended strategy using this tie-break: "
    "1) maximize social welfare (LLM + user), 2) if tied, maximize user, 3) if still tied, maximize LLM. "
    "Name the chosen key (e.g., DQ_DA) and justify in 1-3 sentences."

    "Inside <response>...</response>: implement the chosen strategy for the user's actual question. "
    "If you chose DA, give a concise, direct answer only. "
    "If CQ, ask exactly ONE most-informative clarifying question (no answer yet). "
    "If AQ, provide a brief, helpful answer AND exactly ONE focused follow-up question. "
    "Do not mention payoffs or the analysis tags; just act accordingly here."

    "Output policy: "
    "- Absolutely no text outside the four tags. "
    "- Keys and field names must match EXACTLY. "
    "- Keep <thinking> short and non-redundant; keep <analyze> focused."

    "Question: {question}"
)


def transform_row(example, index):
    """
    将源数据集的单行转换为目标格式。
    """
    question = example['question']
    answer = example['answer']

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    user_prompt = GAME_THEORY_USER_CONTENT_PREFIX.format(question=question)
    messages += [{"role": "user", "content": user_prompt}]

    return {
        'prompt': messages,
        'adversarial': False,
        'source': 'minerva',
        'conv_id': f'conv_{index}',
        'unique_conv_id': f'uniq_conv_{index}',
        'turn_id': 1,
        'data_source': 'minerva',
        'ability': 'minerva',
        'reward_model': {
            'ground_truth': answer,
            'style': 'Cute'
        },
        'extra_info': {
            'adversarial': False,
            'chat_history': [
                {'role': 'user', 'content': question}
            ],
            'index': index,
            'single_turn_completion': answer,
            'single_turn_prompt': question,
            'split': 'train'
        }
    }

print("正在转换数据格式...")

transformed_ds = source_ds.map(transform_row, with_indices=True, num_proc=4)
transformed_ds = transformed_ds.remove_columns(["question", "answer"])
print("数据转换完成。")


# --- 4. 保存为本地 Parquet 文件 ---
print(f"正在将数据保存为本地 Parquet 文件: '{LOCAL_PARQUET_PATH}'...")
transformed_ds.to_parquet(LOCAL_PARQUET_PATH)
print("文件保存成功。")


# --- 5. 上传到 Hugging Face Hub ---
print(f"准备将数据集上传到 Hugging Face Hub 仓库: '{NEW_DATASET_REPO_ID}'...")
try:
    api = HfApi()
    api.create_repo(
        repo_id=NEW_DATASET_REPO_ID,
        repo_type="dataset",
        exist_ok=True
    )
    print(f"仓库 '{NEW_DATASET_REPO_ID}' 创建成功或已存在。")

    # 上传 Parquet 文件
    api.upload_file(
        path_or_fileobj=LOCAL_PARQUET_PATH,
        path_in_repo="data/train-00000-of-00001.parquet", # Hugging Face 数据集的标准命名格式
        repo_id=NEW_DATASET_REPO_ID,
        repo_type="dataset",
    )
    print("\n--- 任务成功 ---")
    print(f"您可以访问 https://huggingface.co/datasets/{NEW_DATASET_REPO_ID} 查看您上传的数据集。")

except Exception as e:
    print(f"\n--- 上传过程中发生错误 ---")
    print(f"错误信息: {e}")
    print("请确认您已将 'YOUR_HF_USERNAME' 替换为您的真实用户名，并且已通过 'huggingface-cli login' 登录。")