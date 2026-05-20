"""
visualization.py - 期权可视化模块

绘制两类图表：
    1. plot_option_price() : 期权价格随标的价格变化曲线（含内在价值、时间价值分解）
    2. plot_greeks()       : 全希腊字母（Delta/Gamma/Vega/Theta/Rho）随标的价格变化曲线

图表自动保存为 PNG 文件，同时弹出交互窗口，适配 GitHub 项目展示。

技术栈：matplotlib（可视化）、numpy（数值计算）

作者：GitHub Portfolio Project
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from bs_pricing import bs_price
from greeks_calc import calc_delta, calc_gamma, calc_vega, calc_theta, calc_rho

# ── 中文字体配置（兼容 Windows/macOS/Linux）
matplotlib.rcParams["font.sans-serif"] = [
    "SimHei", "Microsoft YaHei", "PingFang SC", "Arial Unicode MS", "DejaVu Sans"
]
matplotlib.rcParams["axes.unicode_minus"] = False  # 修复负号显示为方块的问题


# ─────────────────────────────────────────────────────────
# 期权价格曲线
# ─────────────────────────────────────────────────────────

def plot_option_price(flag, K, T, r, sigma, q=0.0, S_range=None, save_path="option_price_curve.png"):
    """
    绘制期权价格随标的价格变化曲线

    图表包含三层信息：
        1. BSM 理论价格曲线（蓝色实线）
        2. 内在价值曲线（红色虚线）—— max(S-K, 0) 看涨 / max(K-S, 0) 看跌
        3. 时间价值区域（蓝色填充）—— BSM价格 - 内在价值

    参数：
        flag      (str)  : 'c' 看涨 / 'p' 看跌
        K         (float): 行权价
        T         (float): 到期期限（年）
        r         (float): 无风险利率（小数）
        sigma     (float): 波动率（小数）
        q         (float): 连续分红率（小数），默认 0
        S_range   (list) : 标的价格区间 [S_min, S_max]，默认 [0.6K, 1.4K]
        save_path (str)  : 图片保存路径，默认 option_price_curve.png
    """
    if S_range is None:
        S_range = [0.6 * K, 1.4 * K]

    # 在价格区间内均匀取 200 个点（足够平滑）
    S_values = np.linspace(S_range[0], S_range[1], 200)

    # 计算各标的价格下的 BSM 期权价格
    prices = np.array([bs_price(flag, s, K, T, r, sigma, q) for s in S_values])

    # 计算内在价值（不折现，反映到期时价值）
    if flag == "c":
        intrinsic = np.maximum(S_values - K, 0)
        option_name = "Call Option"
    else:
        intrinsic = np.maximum(K - S_values, 0)
        option_name = "Put Option"

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot BSM theoretical price curve
    ax.plot(S_values, prices, color="steelblue", linewidth=2.5, label="BSM Theoretical Price")

    # Plot intrinsic value curve
    ax.plot(S_values, intrinsic, color="crimson", linewidth=1.8,
            linestyle="--", label="Intrinsic Value")

    # Fill time value region (area between BSM price and intrinsic value)
    ax.fill_between(
        S_values, intrinsic, prices,
        alpha=0.18, color="steelblue", label="Time Value"
    )

    # Mark strike price vertical line
    ax.axvline(x=K, color="gray", linestyle=":", linewidth=1.5, label=f"Strike K = {K}")

    ax.set_xlabel("Underlying Price S", fontsize=12)
    ax.set_ylabel("Option Price", fontsize=12)
    ax.set_title(
        f"{option_name} — Price Curve\n"
        f"K={K}  T={T}yr  r={r*100:.1f}%  σ={sigma*100:.1f}%  q={q*100:.1f}%",
        fontsize=13
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  → Option price curve saved to {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────
# 希腊字母曲线
# ─────────────────────────────────────────────────────────

def plot_greeks(flag, K, T, r, sigma, q=0.0, S_range=None, save_path="greeks_curve.png"):
    """
    绘制全希腊字母随标的价格变化曲线（2×3 子图布局）

    图表包含 5 个子图：
        Delta / Gamma / Vega / Theta / Rho
    每个子图均标注行权价位置（灰色虚线）及零轴（黑色细线）。

    参数：
        flag      (str)  : 'c' 看涨 / 'p' 看跌
        K         (float): 行权价
        T         (float): 到期期限（年）
        r         (float): 无风险利率（小数）
        sigma     (float): 波动率（小数）
        q         (float): 连续分红率（小数），默认 0
        S_range   (list) : 标的价格区间 [S_min, S_max]，默认 [0.6K, 1.4K]
        save_path (str)  : 图片保存路径，默认 greeks_curve.png
    """
    if S_range is None:
        S_range = [0.6 * K, 1.4 * K]

    S_values = np.linspace(S_range[0], S_range[1], 200)

    # 批量计算各希腊字母（向量化）
    deltas = np.array([calc_delta(flag, s, K, T, r, sigma, q) for s in S_values])
    gammas = np.array([calc_gamma(flag, s, K, T, r, sigma, q) for s in S_values])
    vegas  = np.array([calc_vega(flag, s, K, T, r, sigma, q)  for s in S_values])
    thetas = np.array([calc_theta(flag, s, K, T, r, sigma, q) for s in S_values])
    rhos   = np.array([calc_rho(flag, s, K, T, r, sigma, q)   for s in S_values])

    option_name = "Call Option" if flag == "c" else "Put Option"

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        f"{option_name} — Greeks Curves\n"
        f"K={K}  T={T}yr  r={r*100:.1f}%  σ={sigma*100:.1f}%  q={q*100:.1f}%",
        fontsize=14
    )

    # Greek subplot configs: (axis, data, name, color, description)
    greek_configs = [
        (axes[0, 0], deltas, "Delta", "steelblue",  "Direction Risk  ∂V/∂S"),
        (axes[0, 1], gammas, "Gamma", "darkorange",  "Delta Convexity  ∂²V/∂S²"),
        (axes[0, 2], vegas,  "Vega",  "forestgreen", "Vol Sensitivity  ∂V/∂σ"),
        (axes[1, 0], thetas, "Theta", "crimson",     "Time Decay  ∂V/∂t"),
        (axes[1, 1], rhos,   "Rho",   "purple",      "Rate Sensitivity  ∂V/∂r"),
    ]

    for ax, data, name, color, desc in greek_configs:
        ax.plot(S_values, data, color=color, linewidth=2)
        # Mark strike price vertical line
        ax.axvline(x=K, color="gray", linestyle=":", linewidth=1, alpha=0.7)
        # Mark zero line
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.6, alpha=0.4)
        ax.set_xlabel("Underlying Price S", fontsize=9)
        ax.set_ylabel(name, fontsize=10)
        ax.set_title(f"{name}  —  {desc}", fontsize=10)
        ax.grid(True, alpha=0.3)

    # Hide the unused 6th subplot slot (2×3 grid, only 5 Greeks)
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  → Greeks curve saved to {save_path}")
    plt.show()
