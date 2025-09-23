#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cd_vs_others_table.py

Row (fixed): cobb-douglas
Columns (opponents): user, llm, linear
Metrics (3 columns): epsilon-pareto coverage advantage, hypervolume advantage, regret advantage

Advantage definition (row better if positive):
- eps_adv  = eps_cov(CD) - eps_cov(opponent)
- hv_adv   = HV(CD) - HV(opponent)
- reg_adv  = regret(opponent) - regret(CD)   # regret is "lower is better", so flip
"""

import os
import csv
from typing import Dict
import numpy as np

# 如果你的模块名是 pareto_compare_plus，就改成那个
from pareto_ratio import load_jsonl, compute_metrics

# ---------------------- CONFIG ----------------------
# METHODS = {
#     "cobb-douglas": "/mnt/bn/heheda/verl/0910_math_cobb/125.jsonl",
#     "user":         "/mnt/bn/heheda/verl/0910_math_user/125.jsonl",
#     "llm":          "/mnt/bn/heheda/verl/0910_math_llm/125.jsonl",
#     "linear":       "/mnt/bn/heheda/verl/0910_math_linear/125.jsonl",
# }

METHODS = {
    "cobb-douglas": "/mnt/bn/heheda/verl/0911_wildguard_cobb/35.jsonl",
    "user":         "/mnt/bn/heheda/verl/0911_wildguard_user/35.jsonl",
    "llm":          "/mnt/bn/heheda/verl/0911_wildguard_llm/35.jsonl",
    "linear":       "/mnt/bn/heheda/verl/0911_wildguard_linear/35.jsonl",
}

# METHODS = {
#     "cobb-douglas": "/mnt/bn/heheda/verl/0910_medium_cobb/85.jsonl",
#     "user":         "/mnt/bn/heheda/verl/0910_medium_user/85.jsonl",
#     "llm":          "/mnt/bn/heheda/verl/0910_medium_llm/85.jsonl",
#     "linear":       "/mnt/bn/heheda/verl/0910_medium_linear/85.jsonl",
# }

# METHODS = {
#     "cobb-douglas": "/mnt/bn/heheda/verl/0910_wildguard_cobb/130.jsonl",
#     "user":         "/mnt/bn/heheda/verl/0910_wildguard_user/130.jsonl",
#     "llm":          "/mnt/bn/heheda/verl/0910_wildguard_llm/130.jsonl",
#     "linear":       "/mnt/bn/heheda/verl/0910_wildguard_linear/130.jsonl",
# }

OPPONENTS = ["user", "llm", "linear"]  # 横轴顺序
EPSILON = 0.1
ALPHA   = 0.5

OUTDIR = "./tables"
OUTCSV = os.path.join(OUTDIR, "cd_vs_others_metrics.csv")
os.makedirs(OUTDIR, exist_ok=True)
# ----------------------------------------------------

def one_vs_one_metrics(path_cd: str, path_opponent: str,
                       eps: float = EPSILON, alpha: float = ALPHA) -> Dict[str, float]:
    """Return advantages of CD over opponent on (eps, hv, regret)."""
    json_cd = load_jsonl(path_cd)
    json_op = load_jsonl(path_opponent)
    m = compute_metrics(json_cd, json_op, key_field="input",
                        reward_keys=('user_reward', 'llm_reward'),
                        eps=eps, alpha=alpha)
    eps_cd = float(m["epsilon_coverage"]["file1"])
    eps_op = float(m["epsilon_coverage"]["file2"])
    reg_cd = float(m["avg_regret_Linf"]["file1"])
    reg_op = float(m["avg_regret_Linf"]["file2"])
    hv_cd  = float(m["hypervolume_ref_(u_min,l_min)"]["file1"])
    hv_op  = float(m["hypervolume_ref_(u_min,l_min)"]["file2"])

    eps = 1e-10
    return {
        "epsilon-pareto coverage": (eps_cd, eps_op),
        "hypervolume": (hv_cd , hv_op),
        "regret advantage": (reg_cd , reg_op)  # flip so positive favors CD
    }

def print_table(rows):
    """Pretty print as a simple text table."""
    headers = ["opponent",
               "epsilon-pareto coverage",
               "hypervolume",
               "regret advantage"]
    # column widths
    w0 = max(len(h) for h in [headers[0]] + [r["opponent"] for r in rows])
    def fmt(x):
        if isinstance(x, tuple):
            return f"({x[0]:.6f}, {x[1]:.6f})"
        return f"{x:.6f}"
    w1 = max(len(headers[1]), max(len(fmt(r["epsilon-pareto coverage"])) for r in rows))
    w2 = max(len(headers[2]), max(len(fmt(r["hypervolume"])) for r in rows))
    w3 = max(len(headers[3]), max(len(fmt(r["regret advantage"])) for r in rows))

    line = f"| {{:<{w0}}} | {{:>{w1}}} | {{:>{w2}}} | {{:>{w3}}} |"
    sep  = f"|{'-'*(w0+2)}|{'-'*(w1+2)}|{'-'*(w2+2)}|{'-'*(w3+2)}|"

    print(sep)
    print(line.format(headers[0], headers[1], headers[2], headers[3]))
    print(sep)
    for r in rows:
        print(line.format(
            r["opponent"],
            fmt(r["epsilon-pareto coverage"]),
            fmt(r["hypervolume"]),
            fmt(r["regret advantage"])
        ))
    print(sep)

def save_csv(rows, path):
    headers = ["opponent", "epsilon-pareto coverage", "hypervolume", "regret advantage"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

def main():
    cd_path = METHODS["cobb-douglas"]
    results = []
    for op in OPPONENTS:
        print(f"CD vs {op}, epsilon={EPSILON}, dataset={cd_path.split('/')[-2]}")
        res = one_vs_one_metrics(cd_path, METHODS[op], eps=EPSILON, alpha=ALPHA)
        results.append({
            "opponent": op,
            **res
        })

    print_table(results)
    save_csv(results, OUTCSV)
    print(f"\nSaved CSV -> {OUTCSV}")

if __name__ == "__main__":
    main()