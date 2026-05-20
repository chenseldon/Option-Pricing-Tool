"""
greeks_calc.py - 期权希腊字母计算模块

基于 Black-Scholes-Merton 解析公式，计算欧式期权的全希腊字母：
    Delta、Gamma、Vega、Theta、Rho

适配场外衍生品敞口监控工作场景：
    - Delta 对冲：做市商实时对冲方向性风险的核心依据
    - Gamma/Vega：波动率风险敞口的量化指标
    - Theta：时间价值损耗监控
    - Rho：利率敏感性分析

技术栈：py_vollib（解析法希腊字母）、pandas（输出汇总表）

作者：GitHub Portfolio Project
"""

import pandas as pd
from py_vollib.black_scholes_merton.greeks.analytical import (
    delta, gamma, vega, theta, rho
)
from bs_pricing import validate_params


# ─────────────────────────────────────────────────────────
# 单项希腊字母计算函数
# ─────────────────────────────────────────────────────────

def calc_delta(flag, S, K, T, r, sigma, q=0.0):
    """
    计算 Delta：∂V/∂S（期权价格对标的价格的一阶偏导）

    BSM 解析公式：
        Call Delta = e^(-qT) · N(d1)         ∈ [0, 1]
        Put  Delta = e^(-qT) · [N(d1) - 1]  ∈ [-1, 0]

    业务含义：
        Delta = 0.5 表示标的价格上涨 1 元，期权价格上涨约 0.5 元。
        场外期权做市商通过买卖标的资产进行 Delta 动态对冲（Delta Hedging），
        是最核心的日常风险管理操作。

    参数：
        flag  (str)  : 'c' 看涨 / 'p' 看跌
        S     (float): 标的价格
        K     (float): 行权价
        T     (float): 到期期限（年）
        r     (float): 无风险利率
        sigma (float): 波动率
        q     (float): 连续分红率，默认 0

    返回：
        float: Delta 值
    """
    valid, msg = validate_params(S, K, T, r, sigma, q)
    if not valid:
        raise ValueError(f"参数校验失败：{msg}")
    return delta(flag, S, K, T, r, sigma, q)


def calc_gamma(flag, S, K, T, r, sigma, q=0.0):
    """
    计算 Gamma：∂²V/∂S²（Delta 对标的价格的偏导，即凸性）

    BSM 解析公式（Call 与 Put 相同）：
        Gamma = e^(-qT) · N'(d1) / (S · σ · √T)
        其中 N'(·) 为标准正态概率密度函数（PDF）

    业务含义：
        Gamma 衡量 Delta 的变化速率。Gamma 越大，动态对冲频率越高，
        对冲成本越大。平值期权 Gamma 最大，深度实/虚值期权 Gamma 趋近 0。

    返回：
        float: Gamma 值（始终 >= 0）
    """
    valid, msg = validate_params(S, K, T, r, sigma, q)
    if not valid:
        raise ValueError(f"参数校验失败：{msg}")
    return gamma(flag, S, K, T, r, sigma, q)


def calc_vega(flag, S, K, T, r, sigma, q=0.0):
    """
    计算 Vega：∂V/∂σ（期权价格对波动率的偏导）

    BSM 解析公式（Call 与 Put 相同）：
        Vega = S · e^(-qT) · N'(d1) · √T

    业务含义：
        Vega 反映波动率变化 1% 对期权价格的影响。
        场外期权做市商需持续监控并对冲 Vega 敞口（即波动率风险），
        通常通过交易场内期权来对冲 Vega。
        py_vollib 输出的 Vega 为波动率变动 1% 时的价格变动量。

    返回：
        float: Vega 值（始终 >= 0）
    """
    valid, msg = validate_params(S, K, T, r, sigma, q)
    if not valid:
        raise ValueError(f"参数校验失败：{msg}")
    return vega(flag, S, K, T, r, sigma, q)


def calc_theta(flag, S, K, T, r, sigma, q=0.0):
    """
    计算 Theta：∂V/∂t（期权价格随时间流逝的变化率）

    业务含义：
        Theta 通常为负值，表示期权的时间价值每天自然衰减。
        期权买方承受 Theta 损耗（持仓成本），期权卖方从时间价值中获益。
        越临近到期日，平值期权 Theta 绝对值越大（时间价值加速衰减）。
        py_vollib 输出的是每日 Theta（已除以 365）。

    返回：
        float: Theta 值（通常 < 0，表示每日损耗）
    """
    valid, msg = validate_params(S, K, T, r, sigma, q)
    if not valid:
        raise ValueError(f"参数校验失败：{msg}")
    return theta(flag, S, K, T, r, sigma, q)


def calc_rho(flag, S, K, T, r, sigma, q=0.0):
    """
    计算 Rho：∂V/∂r（期权价格对无风险利率的偏导）

    BSM 解析公式：
        Call Rho =  K · T · e^(-rT) · N(d2)   （> 0）
        Put  Rho = -K · T · e^(-rT) · N(-d2)  （< 0）

    业务含义：
        Rho 反映无风险利率变化 1% 对期权价格的影响。
        股指期权中 Rho 影响相对较小，但在利率期权或长期限期权中不可忽视。
        py_vollib 输出的 Rho 为利率变动 1% 时的价格变动量。

    返回：
        float: Rho 值
    """
    valid, msg = validate_params(S, K, T, r, sigma, q)
    if not valid:
        raise ValueError(f"参数校验失败：{msg}")
    return rho(flag, S, K, T, r, sigma, q)


# ─────────────────────────────────────────────────────────
# 全希腊字母汇总
# ─────────────────────────────────────────────────────────

def calc_all_greeks(flag, S, K, T, r, sigma, q=0.0):
    """
    计算并汇总全部希腊字母（Delta / Gamma / Vega / Theta / Rho）

    参数：
        flag  (str)  : 期权类型，'c' 看涨，'p' 看跌
        S     (float): 标的价格
        K     (float): 行权价
        T     (float): 到期期限（年）
        r     (float): 无风险利率（小数）
        sigma (float): 波动率（小数）
        q     (float): 连续分红率（小数），默认 0

    返回：
        pd.DataFrame: 含希腊字母名称、数值、业务含义的汇总 DataFrame
    """
    greeks_data = {
        "希腊字母": ["Delta", "Gamma", "Vega", "Theta", "Rho"],
        "数值": [
            calc_delta(flag, S, K, T, r, sigma, q),
            calc_gamma(flag, S, K, T, r, sigma, q),
            calc_vega(flag, S, K, T, r, sigma, q),
            calc_theta(flag, S, K, T, r, sigma, q),
            calc_rho(flag, S, K, T, r, sigma, q),
        ],
        "业务含义": [
            "标的价格变动 1 元，期权价格变动量（方向性风险）",
            "Delta 随标的价格的变化速率（对冲凸性）",
            "波动率变动 1%，期权价格变动量（波动率风险）",
            "每日时间价值损耗（负值表示买方持仓成本）",
            "无风险利率变动 1%，期权价格变动量（利率风险）",
        ],
    }

    df = pd.DataFrame(greeks_data)
    # 保留 6 位小数，便于阅读
    df["数值"] = df["数值"].round(6)
    return df
