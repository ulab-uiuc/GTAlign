"""
python /mnt/bn/heheda/verl/examples/data_preprocess/abg_coqa/abg_coqa.py \
  --input "/mnt/bn/heheda/verl/examples/data_preprocess/abg_coqa/coqa_abg_test.json" \
  --out_jsonl "/mnt/bn/heheda/verl/examples/data_preprocess/abg_coqa/converted" \
  --repo_id "zsqzz/coqa-abg-chat" \
  --split "test" \
  --private        # 如果想创建私有仓库就加上这个开关
"""


import os
import json
import argparse
from typing import List, Dict, Any, Tuple
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    pd = None


# We import lazily to allow running conversion without internet
def lazy_import_datasets():
    import datasets  # type: ignore
    return datasets




INITIAL_CONTENT_PREFIX = (
    "The user's initial question is"
    "Question: {question}"
)

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



def read_json_any(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
        # Try JSON lines first
        # First try to parse as standard JSON
        try:
            return json.loads(txt)
        except Exception:
            # Fallback: try JSON Lines
            items = []
            for line in txt.splitlines():
                if not line.strip():
                    continue
                items.append(json.loads(line))
            return items

def coqa_like_to_messages(example: Dict[str, Any]) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """
    Convert an Ambig‑CoQA/CoQA‑style entry to a list of chat messages:
    - We stitch the story + (optional) history turns into the user prompt.
    - The assistant content is the target answer.
    Returns: (messages, meta)
    """
    story = example.get("story", "")
    target = example.get("target_turn", {}) or {}
    question = target.get("question", "")
    answer = target.get("answer", "")
    turn_id = target.get("turn_id", None)

    # Build a short context from history turns (if available)
    hist = example.get("history_turns", []) or []
    hist_lines = []
    for ht in hist:
        hq = str(ht.get("question", "")).strip()
        ha = str(ht.get("answer", "")).strip()
        if hq or ha:
            hist_lines.append(f"Q: {hq}\nA: {ha}")
    history_block = ("\n\n".join(hist_lines)).strip()

    user_prompt_parts = []
    if story:
        user_prompt_parts.append(f"Context:\n{story}")
    if history_block:
        user_prompt_parts.append(f"Previous Q&A:\n{history_block}")
    if question:
        user_prompt_parts.append(f"Question:\n{question}")

    user_content = "\n\n".join(user_prompt_parts).strip()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    user_prompt = GAME_THEORY_USER_CONTENT_PREFIX.format(question=user_content)
    
    messages += [{"role": "user", "content": user_prompt}]


    # Meta columns that mirror the screenshot-style schema (with safe defaults)
    meta = {
        "adversarial": bool(example.get("adversarial", False)),
        "source": "abg-coqa",
        "conv_id": example.get("conv_id", None),
        "unique_conv_id": example.get("unique_conv_id", None),
        "turn_id": turn_id,
        "data_source": "abg-coqa",
        "ability": "abg-coqa",
        "reward_model": example.get("reward_model", {"ground_truth": example.get("ambiguity", "non_ambiguous"), "style": "rule"}),
        "extra_info": {"ground_truth": example.get("ambiguity", "non_ambiguous"), "single_turn_prompt": user_content},
    }
    return messages, meta

def to_hf_records(raw: Any) -> List[Dict[str, Any]]:
    """
    Normalize input to a list and convert each element to an HF‑ready record.
    """
    if isinstance(raw, dict) and "data" in raw and isinstance(raw["data"], list):
        items = raw["data"]
    elif isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        # Single example dict
        items = [raw]
    else:
        raise ValueError("Unsupported JSON structure for conversion.")
    
    out: List[Dict[str, Any]] = []
    for idx, ex in enumerate(items):
        messages, meta = coqa_like_to_messages(ex)
        rec = {"prompt": messages}
        rec.update(meta)
        # Provide stable ids if missing
        rec.setdefault("conv_id", idx)
        rec.setdefault("unique_conv_id", idx)
        out.append(rec)
    return out

def save_jsonl(path: str, records: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def push_to_hub(records, repo_id: str, split: str = "train", private: bool = False):
    import datasets
    ds = datasets.Dataset.from_list(records)
    # 不再 cast，避免版本差异导致的 schema 冲突
    dsd = datasets.DatasetDict({split: ds})
    dsd.push_to_hub(repo_id, private=private)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input JSON/JSONL")
    ap.add_argument("--out_jsonl", default="converted.jsonl", help="Path to write converted JSONL")
    ap.add_argument("--repo_id", default=None, help="huggingface repo id, e.g., username/dataset-name")
    ap.add_argument("--split", default="train", help="split name to use on hub")
    ap.add_argument("--private", action="store_true", help="create/update a private repo")
    args = ap.parse_args()

    raw = read_json_any(args.input)
    records = to_hf_records(raw)

    # Save local JSONL
    out_path = Path(args.out_jsonl).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(str(out_path), records)

    # Optional: push to hub if repo_id provided
    if args.repo_id:
        # If HF_TOKEN env var is set, datasets will pick it up. Otherwise, rely on cached login.
        print(f"Pushing {len(records)} records to hub repo: {args.repo_id} (split={args.split}, private={args.private})")
        push_to_hub(records, args.repo_id, split=args.split, private=args.private)
    else:
        print(f"Converted {len(records)} records. Wrote JSONL to: {out_path}")

    # Optional preview as CSV if pandas available
    # Optional preview as CSV/Parquet if pandas available
    if True:
        try:
            import pandas as pd
            df = pd.DataFrame(records)
            preview_path = out_path.with_suffix(".preview.csv")
            keep_cols = ["prompt", "adversarial", "source", "conv_id", "unique_conv_id", "turn_id", "data_source", "ability"]
            df[keep_cols].head(20).to_csv(preview_path, index=False)
            print(f"Wrote preview CSV to: {preview_path}")

            # --- 新增：保存完整数据为 parquet ---
            parquet_path = out_path.with_suffix(".parquet")
            df.to_parquet(parquet_path, index=False)
            print(f"Wrote full dataset to Parquet: {parquet_path}")

        except Exception as e:
            print("Preview/Parquet generation skipped:", e)

if __name__ == "__main__":
    main()
