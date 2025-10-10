import json
import math
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

def load_jsonl(path: str) -> List[Dict]:
    out: List[Dict] = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out

def safe_div(x, y):
    return x / y if y else 0.0

def precision_recall_f1(tp, fp, fn):
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1

def tied_rank(scores: List[float]) -> List[float]:
    """返回含 ties 的平均秩次（1-based）。"""
    sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(scores):
        j = i
        # 找到相同分数的区间 [i, j)
        while j < len(scores) and scores[sorted_idx[j]] == scores[sorted_idx[i]]:
            j += 1
        # 区间平均秩
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[sorted_idx[k]] = avg_rank
        i = j
    return ranks

if __name__ == "__main__":
    # path1 = 'verl/0910_total_base/0.jsonl'
    # path1 = 'verl/0912_total_sft/0.jsonl'
    # path1 = "verl/0910_abgqa_linear/140.jsonl"
    path1 = "verl/0910_abgqa_user/130.jsonl"

    json1 = load_jsonl(path1)

    print(json1[0]["answer_reward"])

    # 统计量
    tp = tn = fp = fn = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)

    y_true = []
    y_pred = []

    for idx, item in enumerate(json1):
        if item["source"] != "ambig_qa":
            continue
        # print(item)
        gts = item["gts"]  # "unharmful" or "harmful"
        reward = int(round(float(item.get("answer_reward", 0))))

        # ground truth 标签：harmful=1，unharmful=0
        gt = 1 if gts.startswith("singleAnswer") else 0

        # 预测：根据 reward 含义进行翻转
        # 对 unharmful：reward=1 表示预测为正确回答 → pred=0 (unharmful)
        # 对 harmful： reward=0 表示预测为正确回答 → pred=1 (harmful)
        if gts.startswith("multipleQAs"):
            pred = 0 if reward == 1 else 1
        else:  # gts == "harmful"
            pred = 1 if reward == 1 else 0

        y_true.append(gt)
        y_pred.append(pred)

        # 混淆矩阵
        if gt == 1 and pred == 1:
            tp += 1
        elif gt == 0 and pred == 0:
            tn += 1
        elif gt == 0 and pred == 1:
            fp += 1
        elif gt == 1 and pred == 0:
            fn += 1

        # 分类别 accuracy 统计
        cls = "singleAnswer" if gt == 1 else "multipleQAs"
        per_class_total[cls] += 1
        if gt == pred:
            per_class_correct[cls] += 1
    # 各类 accuracy
    acc_harmful = safe_div(per_class_correct["singleAnswer"], per_class_total["singleAnswer"])
    acc_unharmful = safe_div(per_class_correct["multipleQAs"], per_class_total["multipleQAs"])

    # 总体 accuracy
    overall_acc = safe_div(tp + tn, tp + tn + fp + fn)

    # F1（以 harmful 为正类）
    prec_pos, rec_pos, f1_pos = precision_recall_f1(tp, fp, fn)

    # 同时给出 negative 类的 F1 以及 macro-F1（可选）
    # 将 unharmful 当作正类时：TP'=TN, FP'=FN, FN'=FP
    prec_neg, rec_neg, f1_neg = precision_recall_f1(tn, fn, fp)
    macro_f1 = (f1_pos + f1_neg) / 2.0

    print("=== Metrics ===")
    print(f"Class accuracy - singleAnswer:   {acc_harmful:.6f}  (correct={per_class_correct['singleAnswer']}, total={per_class_total['singleAnswer']})")
    print(f"Class accuracy - multipleQAs: {acc_unharmful:.6f}  (correct={per_class_correct['multipleQAs']}, total={per_class_total['multipleQAs']})")
    print(f"Overall accuracy:           {overall_acc:.6f}")
    print(f"F1 (singleAnswer as positive):   {f1_pos:.6f}  (precision={prec_pos:.6f}, recall={rec_pos:.6f})")
    print(f"Macro-F1 (singleAnswer/multipleQAs): {macro_f1:.6f}  [F1_singleAnswer={f1_pos:.6f}, F1_multipleQAs={f1_neg:.6f}]")
