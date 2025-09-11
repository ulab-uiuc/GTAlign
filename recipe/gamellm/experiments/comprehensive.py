import numpy as np
from itertools import product

# -------- Pareto 工具 --------
def is_dominated(a, b):
    """a 是否被 b 支配"""
    return all(b >= a) and any(b > a)

def pareto_front(points):
    front = []
    for i, p in enumerate(points):
        if not any(is_dominated(p, q) for j, q in enumerate(points) if i != j):
            front.append(p)
    return np.array(front)

# -------- Hypervolume --------
def hypervolume(front, ref):
    """只适合 2D：计算矩形面积并集"""
    f = front[front[:,0].argsort()]  # 按 user_reward 排序
    hv, prev_u = 0, ref[0]
    for u, m in f[::-1]:
        hv += (prev_u - u) * (ref[1] - m)
        prev_u = u
    return hv

# -------- ε-Indicator --------
def epsilon_indicator(A, B):
    """A 相对 B 的 ε 指标 (additive)"""
    eps = -np.inf
    for a in A:
        eps = max(eps, min(max(b - a) for b in B))
    return eps

# -------- Scalarization Regret --------
def scalarization_regret(points, all_points, n_pref=1000):
    regrets = []
    for _ in range(n_pref):
        w = np.random.dirichlet([1,1])
        vals = [np.dot(w,p) for p in points]
        opt = max(np.dot(w,p) for p in all_points)
        regrets.append(opt - max(vals))
    return np.mean(regrets), np.percentile(regrets,[50,90])

# -------- Constrained AUC --------
def constrained_auc(points, all_points, taus, mode="llm"):
    vals = []
    for tau in taus:
        if mode == "llm":
            feas = [p for p in points if p[1] >= tau]
            opt = max((p[0] for p in feas), default=0)
            ref = max((p[0] for p in all_points if p[1] >= tau), default=0)
        else:
            feas = [p for p in points if p[0] >= tau]
            opt = max((p[1] for p in feas), default=0)
            ref = max((p[1] for p in all_points if p[0] >= tau), default=0)
        vals.append(opt/ref if ref>0 else 0)
    return np.mean(vals)

# ================== 使用示例 ==================
# points_method = np.array([[ru1, rm1], [ru2, rm2], ...])
# points_baseline = ...
# ref_point = np.array([0,0])

# 1. Pareto 前沿
# front = pareto_front(points_method)

# 2. Hypervolume
# hv = hypervolume(front, ref_point)

# 3. ε-Indicator
# eps = epsilon_indicator(front, pareto_front(points_baseline))

# 4. Scalarization Regret
# mean_reg, (p50, p90) = scalarization_regret(front, np.vstack([points_method, points_baseline]))

# 5. Constrained AUC
# auc = constrained_auc(front, np.vstack([points_method, points_baseline]), taus=np.linspace(0,1,20))