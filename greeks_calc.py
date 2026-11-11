"""greeks_calc.py - 期权希腊字母计算模块（兼容层）

解析解实现已迁移至自研的 :mod:`otcpricer.models.bsm`，不再依赖 py_vollib。
本文件保留为**兼容层**，函数签名、单位约定与输出表格结构一字不改，
``app.py`` / ``main.py`` / ``visualization.py`` 无需改动。

单位约定（沿用原 py_vollib 口径，与交易台读法一致）：
    - Delta : 标的变动 1 个单位
    - Gamma : 标的变动 1 个单位的平方
    - Vega  : 波动率变动 **1 个百分点**
    - Theta : **每个自然日**
    - Rho   : 利率变动 **1 个百分点**

新代码请直接用 :func:`otcpricer.models.bsm.greeks`，它一次性返回五个希腊字母
且只算一遍 d1/d2；本文件的逐个函数各自重算一次，是为了保持旧接口。
"""

from __future__ import annotations

import pandas as pd

from otcpricer.models import bsm

__all__ = [
    "calc_all_greeks",
    "calc_delta",
    "calc_gamma",
    "calc_rho",
    "calc_theta",
    "calc_vega",
]


# ─────────────────────────────────────────────────────────
# 单项希腊字母计算函数
# ─────────────────────────────────────────────────────────


def calc_delta(flag: str, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0):
    """Delta：``∂V/∂S``（期权价格对标的价格的一阶偏导）。

    ``Call Delta = e^(-qT)·N(d1) ∈ [0, 1]``；
    ``Put  Delta = -e^(-qT)·N(-d1) ∈ [-1, 0]``。

    业务含义：做市商据此买卖标的做 Delta 动态对冲，是最核心的日常风险管理操作。
    支持 numpy 数组入参，用于一次性算出整条 Delta 曲线。
    """
    return bsm.delta(flag, S, K, T, r, sigma, q)


def calc_gamma(flag: str, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0):
    """Gamma：``∂²V/∂S² = e^(-qT)·N'(d1) / (S·σ·√T)``，看涨看跌相同。

    业务含义：Delta 的变化速率。Gamma 越大，动态对冲越频繁、对冲成本越高。
    平值期权 Gamma 最大，深度实/虚值趋近 0。
    """
    return bsm.gamma(flag, S, K, T, r, sigma, q)


def calc_vega(flag: str, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0):
    """Vega：``∂V/∂σ = S·e^(-qT)·N'(d1)·√T``，按波动率变动 1 个百分点计价。

    业务含义：波动率风险敞口。做市商通常通过交易场内期权来对冲 Vega。
    """
    return bsm.vega(flag, S, K, T, r, sigma, q)


def calc_theta(flag: str, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0):
    """Theta：``∂V/∂t``，按**每个自然日**计价（已除以 365）。

    业务含义：通常为负，表示时间价值的自然衰减 —— 买方的持仓成本、卖方的收入。
    越临近到期，平值期权的 Theta 绝对值越大。
    """
    return bsm.theta(flag, S, K, T, r, sigma, q)


def calc_rho(flag: str, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0):
    """Rho：``∂V/∂r``，按利率变动 1 个百分点计价。

    ``Call Rho = T·K·e^(-rT)·N(d2) > 0``；``Put Rho = -T·K·e^(-rT)·N(-d2) < 0``。

    业务含义：股指期权中影响较小，但长期限或深度实值期权不可忽视。
    """
    return bsm.rho(flag, S, K, T, r, sigma, q)


# ─────────────────────────────────────────────────────────
# 全希腊字母汇总
# ─────────────────────────────────────────────────────────


def calc_all_greeks(
    flag: str, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0
) -> pd.DataFrame:
    """计算并汇总全部希腊字母。

    底层走 :func:`otcpricer.models.bsm.greeks`，五个希腊字母共用一次 d1/d2
    计算，而非像逐个函数那样重算五遍。

    返回：
        pd.DataFrame: 含「希腊字母 / 数值 / 业务含义」三列的汇总表，
        数值保留 6 位小数。
    """
    g = bsm.greeks(flag, S, K, T, r, sigma, q)

    greeks_data = {
        "希腊字母": ["Delta", "Gamma", "Vega", "Theta", "Rho"],
        "数值": [g.delta, g.gamma, g.vega, g.theta, g.rho],
        "业务含义": [
            "标的价格变动 1 元，期权价格变动量（方向性风险）",
            "Delta 随标的价格的变化速率（对冲凸性）",
            "波动率变动 1%，期权价格变动量（波动率风险）",
            "每日时间价值损耗（负值表示买方持仓成本）",
            "无风险利率变动 1%，期权价格变动量（利率风险）",
        ],
    }

    df = pd.DataFrame(greeks_data)
    df["数值"] = df["数值"].round(6)
    return df
