#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pairwise_heatmaps_upper_norm.py

- 归一化：对每个指标矩阵分别用其 max(|value|) 做分母，得到相对优势（约在[-1,1]）
- 可视化：仅显示上三角（i<j），对角线与下三角设为 NaN 并透明
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pareto_ratio import load_jsonl, compute_metrics  # 若你的文件名是 pareto_compare_plus.py，请改成相应模块名

# ---------------------- CONFIG ----------------------
METHODS = {
    "cobb-douglas": "verl/0910_wildguard_cobb/140.jsonl",
    "linear":       "verl/0910_wildguard_linear/140.jsonl",
    "user":         "verl/0910_wildguard_user/140.jsonl",
    "llm":          "verl/0910_wildguard_llm/140.jsonl",
}

EPSILON = 0.0
ALPHA   = 0.5

OUTDIR = "./heatmaps"
os.makedirs(OUTDIR, exist_ok=True)
# ----------------------------------------------------

def one_vs_one_metrics(path_a, path_b, eps=EPSILON, alpha=ALPHA):
    """Compute three row-vs-col advantages using compute_metrics."""
    json_a = load_jsonl(path_a)
    json_b = load_jsonl(path_b)
    m = compute_metrics(json_a, json_b, key_field="input",
                        reward_keys=('user_reward', 'llm_reward'),
                        eps=eps, alpha=alpha)
    epsA = float(m["epsilon_coverage"]["file1"])
    epsB = float(m["epsilon_coverage"]["file2"])
    regA = float(m["avg_regret_Linf"]["file1"])
    regB = float(m["avg_regret_Linf"]["file2"])
    hvA  = float(m["hypervolume_ref_(u_min,l_min)"]["file1"])
    hvB  = float(m["hypervolume_ref_(u_min,l_min)"]["file2"])
    return {
        "eps_cov_diff": epsA - epsB,     # higher => row better
        "regret_diff":  regB - regA,     # lower regret better -> flip so higher => row better
        "hv_diff":      hvA - hvB        # higher => row better
    }

def build_adv_matrices(labels, methods):
    """Build raw advantage matrices for all three metrics."""
    n = len(labels)
    eps_mat = np.zeros((n, n), dtype=float)
    reg_mat = np.zeros((n, n), dtype=float)
    hv_mat  = np.zeros((n, n), dtype=float)

    for i, ai in enumerate(labels):
        for j, bj in enumerate(labels):
            if i == j:
                eps_mat[i, j] = 0.0
                reg_mat[i, j] = 0.0
                hv_mat[i, j]  = 0.0
            else:
                res = one_vs_one_metrics(methods[ai], methods[bj], eps=EPSILON, alpha=ALPHA)
                eps_mat[i, j] = res["eps_cov_diff"]
                reg_mat[i, j] = res["regret_diff"]
                hv_mat[i, j]  = res["hv_diff"]
    return eps_mat, reg_mat, hv_mat

def normalize_by_max_abs(M):
    """Normalize matrix by its global max absolute value; keep zeros if matrix is all zeros."""
    max_abs = np.nanmax(np.abs(M))
    if max_abs > 0:
        return M / max_abs
    return M.copy()

def upper_triangle_only(M):
    """Keep only upper triangle (i<j); set others to NaN."""
    U = np.full_like(M, np.nan, dtype=float)
    iu, ju = np.triu_indices_from(M, k=1)  # k=1 excludes diagonal
    U[iu, ju] = M[iu, ju]
    return U

def annotate(ax, data, fmt=".3f"):
    """Put numbers on visible (non-NaN) cells only."""
    n, m = data.shape
    for i in range(n):
        for j in range(m):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, format(val, fmt), ha="center", va="center")

def plot_heatmap_upper(matrix, labels, title, outfile):
    """Plot only upper triangle with normalization already applied."""
    mat = np.array(matrix, dtype=float)
    # Mask NaNs for transparency
    masked = np.ma.array(mat, mask=np.isnan(mat))

    fig, ax = plt.subplots(figsize=(7, 6))
    # Center color around 0 for symmetric advantages
    finite_vals = masked.compressed()
    vmax = np.max(np.abs(finite_vals)) if finite_vals.size > 0 else 1.0
    im = ax.imshow(masked, vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title, pad=10)

    # grid lines to hint upper-tri layout
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_ylim(len(labels) - 0.5, -0.5)

    annotate(ax, mat, fmt=".3f")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized advantage (row vs col)")

    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)

def main():
    labels = list(METHODS.keys())

    # 1) 构建原始三种优势矩阵
    eps_mat_raw, reg_mat_raw, hv_mat_raw = build_adv_matrices(labels, METHODS)

    # 2) 分别做 max-abs 归一化
    eps_mat_norm = normalize_by_max_abs(eps_mat_raw)
    reg_mat_norm = normalize_by_max_abs(reg_mat_raw)
    hv_mat_norm  = normalize_by_max_abs(hv_mat_raw)

    # 3) 仅保留上三角（i<j）
    eps_upper = upper_triangle_only(eps_mat_norm)
    reg_upper = upper_triangle_only(reg_mat_norm)
    hv_upper  = upper_triangle_only(hv_mat_norm)

    # 4) 绘图（仅上三角）
    plot_heatmap_upper(
        eps_upper, labels,
        title=f"ε-Pareto Coverage Advantage (normalized, ε={EPSILON}) — Upper Triangle",
        outfile=os.path.join(OUTDIR, f"heatmap_epscov_upper_norm_eps{EPSILON:.3f}.png"),
    )
    plot_heatmap_upper(
        reg_upper, labels,
        title="Avg L∞ Regret Advantage (normalized) — Upper Triangle",
        outfile=os.path.join(OUTDIR, "heatmap_regret_upper_norm.png"),
    )
    plot_heatmap_upper(
        hv_upper, labels,
        title="Hypervolume Advantage (normalized) — Upper Triangle",
        outfile=os.path.join(OUTDIR, "heatmap_hv_upper_norm.png"),
    )

    print("Saved normalized, upper-triangle heatmaps:")
    print(" -", os.path.join(OUTDIR, f"heatmap_epscov_upper_norm_eps{EPSILON:.3f}.png"))
    print(" -", os.path.join(OUTDIR, "heatmap_regret_upper_norm.png"))
    print(" -", os.path.join(OUTDIR, "heatmap_hv_upper_norm.png"))

if __name__ == "__main__":
    main()