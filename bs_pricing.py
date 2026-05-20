"""
bs_pricing.py - Black-Scholes-Merton 期权定价模块

实现欧式期权的 Black-Scholes-Merton (BSM) 定价模型和蒙特卡洛模拟定价。
支持含连续分红率的完整 BSM 模型，适配场外香草期权定价核心需求。

模块功能：
    - validate_params()        : 输入参数合法性校验
    - bs_price()               : BSM 模型定价（核心）
    - monte_carlo_price()      : 蒙特卡洛模拟定价（辅助验证）
    - calc_implied_volatility(): 隐含波动率反推

技术栈：py_vollib（核心定价库）、numpy（数值计算）

作者：GitHub Portfolio Project
"""

import numpy as np
from py_vollib.black_scholes_merton import black_scholes_merton
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility as _bsm_iv


# ─────────────────────────────────────────────────────────
# 参数校验
# ─────────────────────────────────────────────────────────

def validate_params(S, K, T, r, sigma, q=0.0):
    """
    期权输入参数合法性校验

    参数说明：
        S     (float): 标的资产当前价格，单位：元/点，需 > 0
        K     (float): 行权价格，需 > 0
        T     (float): 到期期限（年化），如 0.25 表示 3 个月，需 > 0
        r     (float): 无风险利率（年化小数），如 0.02 表示 2%
        sigma (float): 年化波动率（小数），合理范围 (0, 2]
        q     (float): 连续分红率（年化小数），需 >= 0，默认为 0

    返回值：
        tuple: (bool, str) —— (是否合法, 错误提示信息)
    """
    if S <= 0:
        return False, f"标的价格 S={S} 无效，必须大于 0"
    if K <= 0:
        return False, f"行权价格 K={K} 无效，必须大于 0"
    if T <= 0:
        return False, f"到期期限 T={T} 无效，必须大于 0（单位：年）"
    if sigma <= 0 or sigma > 2:
        return False, f"波动率 sigma={sigma} 无效，应在 (0, 2] 范围内"
    if q < 0:
        return False, f"分红率 q={q} 无效，必须大于等于 0"
    return True, "参数合法"


# ─────────────────────────────────────────────────────────
# BSM 定价
# ─────────────────────────────────────────────────────────

def bs_price(flag, S, K, T, r, sigma, q=0.0):
    """
    Black-Scholes-Merton (BSM) 模型欧式期权定价

    BSM 公式（含连续分红率 q）：
        看涨 Call：C = S·e^(-qT)·N(d1) - K·e^(-rT)·N(d2)
        看跌 Put ：P = K·e^(-rT)·N(-d2) - S·e^(-qT)·N(-d1)

    其中：
        d1 = [ln(S/K) + (r - q + σ²/2)·T] / (σ·√T)
        d2 = d1 - σ·√T
        N(·) : 标准正态分布累积密度函数（CDF）

    参数：
        flag  (str)  : 期权类型，'c' 看涨（Call），'p' 看跌（Put）
        S     (float): 标的价格
        K     (float): 行权价
        T     (float): 到期期限（年）
        r     (float): 无风险利率（小数）
        sigma (float): 年化波动率（小数）
        q     (float): 连续分红率（小数），默认 0

    返回：
        float: BSM 期权理论价格
    """
    valid, msg = validate_params(S, K, T, r, sigma, q)
    if not valid:
        raise ValueError(f"参数校验失败：{msg}")

    # 调用 py_vollib 内置 BSM 定价函数，封装了 N(d1)/N(d2) 计算
    price = black_scholes_merton(flag, S, K, T, r, sigma, q)
    return price


# ─────────────────────────────────────────────────────────
# 蒙特卡洛模拟定价
# ─────────────────────────────────────────────────────────

def monte_carlo_price(flag, S, K, T, r, sigma, q=0.0, n_simulations=100000, seed=42):
    """
    蒙特卡洛模拟定价（欧式期权，风险中性测度）

    原理：
        在风险中性测度下，标的价格服从几何布朗运动（GBM）：
            S_T = S · exp[(r - q - σ²/2)·T + σ·√T·Z]
        其中 Z ~ N(0,1) 为标准正态随机变量。

        模拟大量到期路径，对到期收益取期望，折现后得到期权价格：
            C = e^(-rT) · E[max(S_T - K, 0)]   （看涨）
            P = e^(-rT) · E[max(K - S_T, 0)]   （看跌）

    参数：
        flag          (str)  : 期权类型，'c' 看涨，'p' 看跌
        S             (float): 标的价格
        K             (float): 行权价
        T             (float): 到期期限（年）
        r             (float): 无风险利率（小数）
        sigma         (float): 年化波动率（小数）
        q             (float): 连续分红率（小数），默认 0
        n_simulations (int)  : 模拟路径数，默认 100000（精度与速度权衡）
        seed          (int)  : 随机数种子，保证结果可复现，默认 42

    返回：
        float: 蒙特卡洛估算的期权价格
    """
    valid, msg = validate_params(S, K, T, r, sigma, q)
    if not valid:
        raise ValueError(f"参数校验失败：{msg}")

    np.random.seed(seed)

    # 生成 n_simulations 个标准正态随机数 Z ~ N(0,1)
    Z = np.random.standard_normal(n_simulations)

    # 风险中性漂移项：(r - q - σ²/2) · T
    drift = (r - q - 0.5 * sigma ** 2) * T

    # 扩散项：σ · √T · Z
    diffusion = sigma * np.sqrt(T) * Z

    # 模拟到期标的价格 S_T（每条路径均为独立终态）
    S_T = S * np.exp(drift + diffusion)

    # 计算各路径期权到期收益（payoff）
    if flag == 'c':
        # 看涨期权：到期收益 = max(S_T - K, 0)
        payoffs = np.maximum(S_T - K, 0)
    elif flag == 'p':
        # 看跌期权：到期收益 = max(K - S_T, 0)
        payoffs = np.maximum(K - S_T, 0)
    else:
        raise ValueError("flag 参数错误，应为 'c'（看涨）或 'p'（看跌）")

    # 对收益折现：e^(-rT) · E[payoff]
    price = np.exp(-r * T) * np.mean(payoffs)
    return price


# ─────────────────────────────────────────────────────────
# 隐含波动率反推
# ─────────────────────────────────────────────────────────

def calc_implied_volatility(flag, market_price, S, K, T, r, q=0.0):
    """
    隐含波动率反推（Implied Volatility, IV）

    原理：
        已知期权市场价格，通过数值迭代（Newton-Raphson 法）反向求解
        BSM 公式，找到使理论价格等于市场价格的波动率，即隐含波动率。

        隐含波动率反映了市场对标的资产未来波动的预期，是场外期权
        定价、波动率曲面构建、风险监控的核心参考指标。

    参数：
        flag         (str)  : 期权类型，'c' 看涨，'p' 看跌
        market_price (float): 期权市场价格（需在无套利范围内）
        S            (float): 标的价格
        K            (float): 行权价
        T            (float): 到期期限（年）
        r            (float): 无风险利率（小数）
        q            (float): 连续分红率（小数），默认 0

    返回：
        float: 隐含波动率（年化小数，如 0.20 表示 20%）

    异常：
        若市场价格不在无套利区间内，或迭代不收敛，将抛出 ValueError。
    """
    if market_price <= 0:
        raise ValueError("市场价格必须大于 0")
    if T <= 0:
        raise ValueError("到期期限必须大于 0")

    try:
        # py_vollib 提供牛顿迭代法求解隐含波动率
        # 参数顺序：(price, S, K, t, r, q, flag)
        iv = _bsm_iv(market_price, S, K, T, r, q, flag)
        return iv
    except Exception as e:
        raise ValueError(
            f"隐含波动率求解失败（市场价格可能超出无套利范围）：{e}"
        )
