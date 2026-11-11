"""bs_pricing.py - Black-Scholes-Merton 定价模块（兼容层）

本模块原先直接依赖 py_vollib。py_vollib 现已是 vollib 的转发层，import 即
抛 DeprecationWarning，且其隐含波动率求解器在深度实值/临近到期的报价上会抛出
含义不清的 BelowIntrinsicException。

定价实现已迁移至自研的 :mod:`otcpricer.models.bsm`：公式逐项可读、全程向量化
（价格曲线端点实测快 ~887 倍）、隐含波动率带 Newton + Brent 双保险与
可识别性筛查。本文件保留为**兼容层**，签名与返回约定一字不改，
``app.py`` / ``main.py`` / ``visualization.py`` 无需改动。

新代码请直接使用 :mod:`otcpricer.models.bsm`。

模块功能：
    - validate_params()        : 输入参数合法性校验（返回 (bool, str)，保持原约定）
    - bs_price()               : BSM 模型定价
    - monte_carlo_price()      : 蒙特卡洛模拟定价（辅助验证）
    - calc_implied_volatility(): 隐含波动率反推
"""

from __future__ import annotations

import numpy as np

from otcpricer.core.exceptions import OtcPricerError, ParameterError
from otcpricer.core.validation import MAX_VOLATILITY, validate_vanilla
from otcpricer.models import bsm

__all__ = [
    "bs_price",
    "calc_implied_volatility",
    "monte_carlo_price",
    "validate_params",
]


# ─────────────────────────────────────────────────────────
# 参数校验
# ─────────────────────────────────────────────────────────


def validate_params(
    S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0
) -> tuple[bool, str]:
    """期权输入参数合法性校验。

    保留原有的 ``(bool, str)`` 返回约定 —— ``main.py`` 的交互模式依赖它。
    库内部新代码改用 :mod:`otcpricer.core.validation`，那里是抛异常而非返回
    标志位，避免调用方忘记检查返回值。

    注意：
        原实现把波动率上限写成 2.0（200%）。这里沿用库里的
        :data:`~otcpricer.core.validation.MAX_VOLATILITY`（500%），因为 200%
        对个股期权与短期限报价来说会误拒合法输入。

    返回：
        tuple: (是否合法, 错误提示信息)
    """
    try:
        validate_vanilla(S, K, T, r, sigma, q)
    except ParameterError as exc:
        return False, str(exc)
    return True, "参数合法"


# ─────────────────────────────────────────────────────────
# BSM 定价
# ─────────────────────────────────────────────────────────


def bs_price(flag: str, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0):
    """Black-Scholes-Merton 模型欧式期权定价。

    看涨 ``C = S·e^(-qT)·N(d1) - K·e^(-rT)·N(d2)``；
    看跌 ``P = K·e^(-rT)·N(-d2) - S·e^(-qT)·N(-d1)``。

    参数：
        flag  : 'c' 看涨 / 'p' 看跌
        S     : 标的价格（支持 numpy 数组，用于一次性算出整条价格曲线）
        K     : 行权价
        T     : 到期期限（年）
        r     : 无风险利率（小数）
        sigma : 年化波动率（小数）
        q     : 连续分红率（小数）

    返回：
        float 或 ndarray: BSM 理论价格。标量入参返回标量。
    """
    return bsm.price(flag, S, K, T, r, sigma, q)


# ─────────────────────────────────────────────────────────
# 蒙特卡洛模拟定价
# ─────────────────────────────────────────────────────────


def monte_carlo_price(
    flag: str,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    n_simulations: int = 100_000,
    seed: int = 42,
) -> float:
    """蒙特卡洛模拟定价（欧式期权，风险中性测度）。

    风险中性测度下标的服从几何布朗运动，到期价格可一步精确抽样：
        ``S_T = S·exp[(r - q - σ²/2)·T + σ·√T·Z]``，``Z ~ N(0,1)``
    期权价值为到期收益的折现期望 ``e^(-rT)·E[payoff]``。

    参数：
        n_simulations : 模拟路径数，默认 100000
        seed          : 随机数种子，保证报价可复现

    返回：
        float: 蒙特卡洛估算价格
    """
    valid, msg = validate_params(S, K, T, r, sigma, q)
    if not valid:
        raise ValueError(f"参数校验失败：{msg}")

    flag_norm = str(flag).lower()
    if flag_norm not in ("c", "p"):
        raise ValueError("flag 参数错误，应为 'c'（看涨）或 'p'（看跌）")

    # 使用独立的 Generator 而非全局 np.random.seed，避免污染调用方的随机状态
    # （原实现会改变进程全局 RNG，使同一请求里后续的抽样结果随调用顺序变化）。
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_simulations)

    drift = (r - q - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * Z
    S_T = S * np.exp(drift + diffusion)

    payoffs = np.maximum(S_T - K, 0.0) if flag_norm == "c" else np.maximum(K - S_T, 0.0)
    return float(np.exp(-r * T) * np.mean(payoffs))


# ─────────────────────────────────────────────────────────
# 隐含波动率反推
# ─────────────────────────────────────────────────────────


def calc_implied_volatility(
    flag: str,
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
) -> float:
    """隐含波动率反推（Newton-Raphson + Brent 兜底）。

    相比原 py_vollib 实现的两点改进：

    1. 报价落在无套利区间之外时，直接给出区间上下界与成因，而不是抛
       ``BelowIntrinsicException`` 这种容易误导的信息。
    2. 深度实值或临近到期的报价，其价格对波动率几乎不敏感（vega 下溢到 0），
       隐含波动率在数值上**不可识别** —— 此时会明确报错，而不是返回一个看似
       合理却毫无信息量的数值去污染后续的波动率曲面拟合。

    参数：
        market_price : 期权市场价格（需在无套利范围内）

    返回：
        float: 隐含波动率（年化小数，0.20 表示 20%）

    异常：
        ValueError: 报价超出无套利区间、迭代不收敛，或该报价无法确定波动率。
    """
    try:
        return bsm.implied_volatility(flag, market_price, S, K, T, r, q)
    except OtcPricerError as exc:
        # 保持原有的 ValueError 契约：旧调用方（含 /api/iv、/api/vol_smile）
        # 按 ValueError 捕获。ParameterError 本身已是 ValueError 的子类。
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"隐含波动率求解失败：{exc}") from exc


# 原实现把波动率上限硬编码为 2.0；这里导出库里的统一上限，便于调用方引用。
MAX_SIGMA = MAX_VOLATILITY
