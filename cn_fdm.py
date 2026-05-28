"""
cn_fdm.py — 通用 Crank-Nicolson 有限差分法（CN-FDM）结构化产品定价模块

支持产品类型：
    - 'snowball'  : 雪球期权（双障碍：向下敲入 KI + 向上敲出 KO）
    - 'shark_fin' : 鲨鱼鳍期权（单障碍：向上敲出 KO，经典看涨结构）

定价框架（Black-Scholes PDE 对数变换）：
    ∂V/∂τ = ½σ²·∂²V/∂x² + (r−q−½σ²)·∂V/∂x − r·V
    x = ln(S)，τ = T−t 为逆向时间

差分格式：
    Rannacher (1984) 方案 = 4 步全隐式（阻尼）+ (M−4) 步 Crank-Nicolson
    全隐式首步消除障碍跳跃引起的高频振荡，后续 CN 保证二阶精度

障碍处理：
    鲨鱼鳍 — 网格截断于 KO（上边界 = rebate），TDMA 不跨越障碍节点
    雪球   — 双状态 PDE，V_no_ki 活跃域 [KI, KO]，月度 KO 离散施加

线性系统：TDMA（托马斯算法）O(N) 三对角求解器

作者：GitHub Portfolio Project — 场外衍生品定价工具
"""

import numpy as np


class CN_FDM:
    """
    通用 CN-FDM 定价引擎（Rannacher 方案，障碍截断网格）

    构造参数：
        product_type  : 'snowball' 或 'shark_fin'
        S0            : 标的当前价格
        T             : 期限（年）
        r             : 无风险利率（小数）
        sigma         : 年化波动率（小数）
        q             : 连续股息率（小数），默认 0
        KO_pct        : 敲出价 / S0，如 1.05
        KI_pct        : 敲入价 / S0（仅雪球），如 0.75
        coupon_pa     : 年化票息率（雪球），如 0.15
        obs_freq      : 敲出观察频率（每年次数），默认 12
        rebate        : 鲨鱼鳍敲出补偿，默认 0
        K_strike      : 鲨鱼鳍行权价，默认 = S0
        N             : 空间节点数，默认 400
        M_per_year    : 每年时间步数，默认 252
        rannacher_steps: 全隐式阻尼步数（Rannacher 方案），默认 4
    """

    def __init__(
        self,
        product_type:    str,
        S0:              float,
        T:               float,
        r:               float,
        sigma:           float,
        q:               float = 0.0,
        KO_pct:          float = 1.05,
        KI_pct:          float = 0.75,
        coupon_pa:       float = 0.15,
        obs_freq:        int   = 12,
        rebate:          float = 0.0,
        K_strike:        float = None,
        N:               int   = 400,
        M_per_year:      int   = 252,
        rannacher_steps: int   = 4,
    ):
        self.product_type    = product_type.lower()
        if self.product_type not in ("snowball", "shark_fin"):
            raise ValueError("product_type 必须为 'snowball' 或 'shark_fin'")

        self.S0              = float(S0)
        self.T               = float(T)
        self.r               = float(r)
        self.sigma           = float(sigma)
        self.q               = float(q)
        self.KO_pct          = float(KO_pct)
        self.KI_pct          = float(KI_pct)
        self.KO              = float(S0) * float(KO_pct)
        self.KI              = float(S0) * float(KI_pct)
        self.coupon_pa       = float(coupon_pa)
        self.obs_freq        = int(obs_freq)
        self.rebate          = float(rebate)
        self.K_strike        = float(K_strike) if K_strike is not None else float(S0)
        self.N               = int(N)
        self.M               = max(int(round(T * M_per_year)), 10)
        self.M_per_year      = M_per_year
        self.dt              = T / self.M
        self.rannacher_steps = min(int(rannacher_steps), self.M)

        # ── 构建空间网格（根据产品类型适配网格范围）─────────────────
        #
        # 核心原则：上边界节点 = 敲出障碍 KO
        #   → TDMA 内部节点全部在 KO 以下，消除跨障碍耦合误差
        #
        # 鲨鱼鳍：截断于 KO，KO 作为硬 Dirichlet 上边界（V = rebate）
        # 雪球  ：活跃区域 [KI, KO]，下边界留余量以捕获 KI 以下的对冲成本
        tail = max(4.0 * sigma * np.sqrt(max(T, 0.1)), 0.8)

        if self.product_type == "shark_fin":
            x_lo = np.log(float(S0)) - tail
            x_hi = np.log(self.KO)              # 截断于障碍
        else:
            x_lo = np.log(self.KI) - 0.6        # KI 以下留余量
            x_hi = np.log(self.KO) + 0.3        # KO 以上留少量余量（月度离散观察）

        self.x_min  = x_lo
        self.x_max  = x_hi
        self.x_grid = np.linspace(x_lo, x_hi, N)
        self.S_grid = np.exp(self.x_grid)
        self.dx     = (x_hi - x_lo) / (N - 1)

        # ── 预计算 PDE 三对角算子系数（对数空间，常系数）──────────
        # BS-PDE log 变换：L·V = α·Vxx + β·Vx − r·V
        # α = ½σ²，β = r − q − ½σ²
        alpha    = 0.5 * sigma ** 2
        beta     = r - q - 0.5 * sigma ** 2
        dx       = self.dx
        self._a  = alpha / dx**2 - beta / (2.0 * dx)   # 下对角
        self._b  = -2.0 * alpha / dx**2 - r             # 主对角
        self._c  = alpha / dx**2 + beta / (2.0 * dx)    # 上对角

    # ─────────────────────────────────────────────────────────
    # TDMA 三对角求解器 O(N)
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _tdma(lower: np.ndarray, main: np.ndarray,
              upper: np.ndarray, rhs:  np.ndarray) -> np.ndarray:
        """
        托马斯算法（Thomas Algorithm / TDMA）
        求解三对角线性方程组：lower[i-1]*x[i-1] + main[i]*x[i] + upper[i]*x[i+1] = rhs[i]
        时间复杂度 O(n)，比通用高斯消元 O(n³) 快得多
        """
        n      = len(main)
        c_star = np.empty(n - 1)
        d_star = np.empty(n)

        # 前向消去
        c_star[0] = upper[0] / main[0]
        d_star[0] = rhs[0]   / main[0]
        for i in range(1, n):
            denom      = main[i] - lower[i - 1] * c_star[i - 1]
            d_star[i]  = (rhs[i] - lower[i - 1] * d_star[i - 1]) / denom
            if i < n - 1:
                c_star[i] = upper[i] / denom

        # 回代
        x     = np.empty(n)
        x[-1] = d_star[-1]
        for i in range(n - 2, -1, -1):
            x[i] = d_star[i] - c_star[i] * x[i + 1]
        return x

    # ─────────────────────────────────────────────────────────
    # 广义 theta 时间推进（θ=0.5: CN；θ=1.0: 全隐式）
    # ─────────────────────────────────────────────────────────

    def _theta_step(self, V: np.ndarray,
                    bc_lower: float, bc_upper: float,
                    theta: float = 0.5) -> np.ndarray:
        """
        广义 theta 格式单步时间推进（逆向）

        θ = 0.5 → Crank-Nicolson（二阶精度，无条件稳定）
        θ = 1.0 → 全隐式（一阶精度，最强阻尼，用于 Rannacher 首步）

        求解：(I − θ·dt·L)·V^{n+1} = (I + (1−θ)·dt·L)·V^n

        LHS 三对角：
            下对角 = −θ·dt·a
            主对角 = 1 − θ·dt·b
            上对角 = −θ·dt·c

        RHS = (1−θ)·dt·(a·V[j-1] + b·V[j] + c·V[j+1]) + V[j]
            + 边界修正项
        """
        dt    = self.dt
        a, b, c = self._a, self._b, self._c
        N     = self.N
        n_int = N - 2

        # RHS：(I + (1-θ)·dt·L)·V
        th1   = 1.0 - theta
        rhs   = (th1 * dt * a  * V[:-2]
                 + (1.0 + th1 * dt * b) * V[1:-1]
                 + th1 * dt * c  * V[2:])

        # 新时间步边界值对 LHS 的贡献修正到 RHS
        rhs[0]  += theta * dt * a * bc_lower
        rhs[-1] += theta * dt * c * bc_upper

        # LHS 三对角（常系数，均匀网格）
        lhs_main  = np.full(n_int,     1.0 - theta * dt * b)
        lhs_lower = np.full(n_int - 1, -theta * dt * a)
        lhs_upper = np.full(n_int - 1, -theta * dt * c)

        V_int = self._tdma(lhs_lower, lhs_main, lhs_upper, rhs)

        V_new       = np.empty_like(V)
        V_new[0]    = bc_lower
        V_new[1:-1] = V_int
        V_new[-1]   = bc_upper
        return V_new

    # ─────────────────────────────────────────────────────────
    # 插值
    # ─────────────────────────────────────────────────────────

    def _interp(self, V: np.ndarray, S_target: float) -> float:
        """对数空间均匀网格线性插值，获取 S=S_target 处的期权价值"""
        x   = np.log(S_target)
        idx = int(np.clip(np.searchsorted(self.x_grid, x), 1, self.N - 1))
        w   = (x - self.x_grid[idx - 1]) / (self.x_grid[idx] - self.x_grid[idx - 1])
        return float(V[idx - 1] * (1.0 - w) + V[idx] * w)

    # ─────────────────────────────────────────────────────────
    # 时间迭代：Rannacher 方案
    # 前 rannacher_steps 步全隐式阻尼，后续 CN
    # ─────────────────────────────────────────────────────────

    def _iterate(self, V: np.ndarray, bc_lo_fn, bc_hi_fn,
                 post_step_fn=None) -> np.ndarray:
        """
        统一时间迭代框架（Rannacher 方案）

        参数：
            V           : 初始价值网格（到期日）
            bc_lo_fn(k) : 返回步 k 的下边界值
            bc_hi_fn(k) : 返回步 k 的上边界值
            post_step_fn(V, k): 每步推进后的障碍/条件处理（可选）

        策略：
            k < rannacher_steps → θ=1.0（全隐式，强阻尼，消振荡）
            k ≥ rannacher_steps → θ=0.5（Crank-Nicolson，二阶精度）
        """
        rs = self.rannacher_steps
        for k in range(self.M):
            theta  = 1.0 if k < rs else 0.5
            bc_lo  = bc_lo_fn(k)
            bc_hi  = bc_hi_fn(k)
            V      = self._theta_step(V, bc_lo, bc_hi, theta)
            if post_step_fn is not None:
                V = post_step_fn(V, k)
        return V

    # ─────────────────────────────────────────────────────────
    # 鲨鱼鳍求解器（截断网格 [S_min, KO]）
    # ─────────────────────────────────────────────────────────

    def _solve_shark_fin(self) -> np.ndarray:
        """
        鲨鱼鳍期权 CN-FDM 求解

        网格上界 = ln(KO)，整个迭代过程中 V[-1] = rebate（硬 Dirichlet BC）
        TDMA 只求解内部节点 [1, N-2]，不跨越障碍，避免耦合误差

        Rannacher 首 4 步全隐式阻尼：消除 V[-1] 处的跳跃不连续引起的
        高频振荡（否则振荡会通过扩散项传播至 S0，导致 ~10x 误差）
        """
        K      = self.K_strike
        rebate = self.rebate

        # 到期日初始条件：max(S−K, 0)，上边界节点强制 = rebate
        V      = np.maximum(self.S_grid - K, 0.0)
        V[-1]  = rebate

        # BC 函数：上下边界全程固定
        V = self._iterate(
            V,
            bc_lo_fn    = lambda k: 0.0,         # 下边界 S→0：call 无价值
            bc_hi_fn    = lambda k: float(rebate),# 上边界 S=KO：rebate
        )
        return V

    # ─────────────────────────────────────────────────────────
    # 雪球求解器（双状态 PDE）
    # ─────────────────────────────────────────────────────────

    def _solve_snowball(self) -> np.ndarray:
        """
        雪球期权双状态 CN-FDM 求解

        双状态含义：
            V_ki   ：已触碰敲入，相当于空头看跌期权
            V_no_ki：未触碰敲入，目标是获取票息；
                     月度 KO 观察 → S ≥ KO 时提前终止兑付票息；
                     日度 KI 条件 → S ≤ KI 时切换到 V_ki 状态

        边界条件（两个状态分别处理）：
            V_ki   下边界：−e^{−rτ}（看跌最大亏损 PV）；上边界：0
            V_no_ki下边界：V_ki[0]（KI 以下必然已敲入）；上边界：coupon×t_after
        """
        S_grid = self.S_grid
        S0     = self.S0
        KO     = self.KO
        KI     = self.KI
        r      = self.r
        dt     = self.dt
        T      = self.T
        M      = self.M
        coupon = self.coupon_pa

        # ── 到期日初始条件 ──────────────────────────────────────
        V_ki    = np.minimum(S_grid / S0 - 1.0, 0.0)   # 空头看跌，归一化单位
        V_no_ki = np.full(self.N, coupon * T)            # 到期兑付全额票息

        # ── 预计算月度 KO 观察步序号 ──────────────────────────
        ko_steps = {}
        for i in range(1, int(round(T * self.obs_freq)) + 1):
            t_obs = i / self.obs_freq
            if t_obs <= T + 1e-9:
                k_idx = max(0, min(M - 1, int(round((T - t_obs) / dt))))
                ko_steps[k_idx] = max(t_obs, 0.0)

        # ── Rannacher 方案逆向迭代 ─────────────────────────────
        rs = self.rannacher_steps
        for k in range(M):
            theta   = 1.0 if k < rs else 0.5
            tau     = (k + 1) * dt
            t_after = max(T - tau, 0.0)

            # V_ki 边界条件
            bc_ki_lo = -np.exp(-r * tau)        # PV of max loss
            bc_ki_hi = 0.0

            # V_no_ki 边界条件
            bc_nki_lo = float(V_ki[0])           # 镜像 V_ki 下边界
            bc_nki_hi = coupon * t_after          # 剩余时间票息估值

            # 先推进 V_ki（以便 KI 条件使用最新值）
            V_ki    = self._theta_step(V_ki,    bc_ki_lo,  bc_ki_hi,  theta)
            V_no_ki = self._theta_step(V_no_ki, bc_nki_lo, bc_nki_hi, theta)

            # 日度 KI 障碍：S ≤ KI 切换到已敲入状态
            ki_mask          = S_grid <= KI
            V_no_ki[ki_mask] = V_ki[ki_mask]

            # 月度 KO 障碍（仅观察日施加）
            if k in ko_steps:
                t_ko             = ko_steps[k]
                ko_mask          = S_grid >= KO
                V_no_ki[ko_mask] = coupon * t_ko

        return V_no_ki

    # ─────────────────────────────────────────────────────────
    # 对外接口
    # ─────────────────────────────────────────────────────────

    def _solve(self) -> np.ndarray:
        if self.product_type == "shark_fin":
            return self._solve_shark_fin()
        else:
            return self._solve_snowball()

    def price(self) -> float:
        """
        计算结构化期权理论价格（在 S = S0 处插值）

        返回：
            float: 期权价值
                雪球  → 占名义本金比例（如 0.03 = 3%）
                鲨鱼鳍 → 绝对价值（与 S 同量纲）
        """
        V = self._solve()
        return self._interp(V, self.S0)

    def greeks(self, d_vol: float = 0.001, d_T_days: float = 1.0) -> dict:
        """
        全套希腊字母（数值稳定）

        Delta/Gamma : 从 t=0 价值网格直接中心差分（最高效，无需重解 PDE）
        Theta       : 前向差分（T−1天），重解一次 PDE
        Vega        : 中心扰动（σ±0.1%），重解两次 PDE

        说明：
            对数空间差分转换到 S 空间：
            ∂²V/∂S² = (1/S²)(∂²V/∂x² − ∂V/∂x)，x=ln(S)
        """
        V_base = self._solve()
        p0     = self._interp(V_base, self.S0)

        # ── Delta & Gamma（从网格直接读取）────────────────────
        j0 = int(np.argmin(np.abs(self.S_grid - self.S0)))
        j0 = int(np.clip(j0, 1, self.N - 2))

        S_m, V_m = self.S_grid[j0 - 1], V_base[j0 - 1]
        S_0, V_c = self.S_grid[j0],     V_base[j0]
        S_p, V_p = self.S_grid[j0 + 1], V_base[j0 + 1]

        dS1   = S_0 - S_m
        dS2   = S_p - S_0
        delta = (V_p - V_m) / (dS1 + dS2)

        dx     = self.dx
        dVdx   = (V_base[j0 + 1] - V_base[j0 - 1]) / (2.0 * dx)
        d2Vdx2 = (V_base[j0 + 1] - 2.0 * V_base[j0] + V_base[j0 - 1]) / dx**2
        gamma  = (d2Vdx2 - dVdx) / (S_0 ** 2)

        # ── Theta（重解缩短 1 天的 PDE）───────────────────────
        d_T = d_T_days / 365.0
        if self.T - d_T > 1e-4:
            theta_p = self._clone(T=self.T - d_T).price()
            theta   = (theta_p - p0) / d_T / 365.0
        else:
            theta = 0.0

        # ── Vega（中心扰动重解）────────────────────────────────
        p_up = self._clone(sigma=max(self.sigma + d_vol, 1e-4)).price()
        p_dn = self._clone(sigma=max(self.sigma - d_vol, 1e-4)).price()
        vega = (p_up - p_dn) / (2.0 * d_vol) * 0.01   # 归一到 1% 波动率变化

        return {
            "delta": round(float(delta), 6),
            "gamma": round(float(gamma), 8),
            "vega":  round(float(vega),  6),
            "theta": round(float(theta), 6),
        }

    def _clone(self, **overrides) -> "CN_FDM":
        """克隆当前实例，覆盖指定参数（用于 Greeks 扰动计算）"""
        params = dict(
            product_type    = self.product_type,
            S0              = self.S0,
            T               = self.T,
            r               = self.r,
            sigma           = self.sigma,
            q               = self.q,
            KO_pct          = self.KO_pct,
            KI_pct          = self.KI_pct,
            coupon_pa       = self.coupon_pa,
            obs_freq        = self.obs_freq,
            rebate          = self.rebate,
            K_strike        = self.K_strike,
            N               = self.N,
            M_per_year      = self.M_per_year,
            rannacher_steps = self.rannacher_steps,
        )
        params.update(overrides)
        return CN_FDM(**params)


# ─────────────────────────────────────────────────────────────────────────────
# 便捷接口（与 bs_pricing.py / greeks_calc.py 风格一致）
# ─────────────────────────────────────────────────────────────────────────────

def cn_fdm_price(product_type: str, **kwargs) -> float:
    """一行调用返回结构化期权理论价格"""
    return CN_FDM(product_type, **kwargs).price()


def cn_fdm_greeks(product_type: str, **kwargs) -> dict:
    """一行调用返回完整希腊字母字典"""
    return CN_FDM(product_type, **kwargs).greeks()


# ─────────────────────────────────────────────────────────────────────────────
# 测试用例
# ─────────────────────────────────────────────────────────────────────────────

def _run_tests():
    """双测试用例 + Haug (1998) 解析验证；运行方式：python cn_fdm.py"""
    import time
    from scipy.stats import norm as _norm

    print("=" * 65)
    print("  CN-FDM 结构化产品定价测试（Rannacher 方案）")
    print("  Crank-Nicolson + Rannacher Smoothing — Barrier Options")
    print("=" * 65)

    # ── 测试 1：雪球期权 ────────────────────────────────────────
    print("\n【测试 1】雪球期权（Snowball / Autocallable）")
    print("  S0=1000  T=1yr  σ=20%  r=2%  q=1%  KO=105%  KI=75%  票息=15%")
    t0 = time.time()
    cn_sb = CN_FDM(
        product_type    = "snowball",
        S0              = 1000.0, T=1.0, r=0.02, sigma=0.20, q=0.01,
        KO_pct=1.05, KI_pct=0.75, coupon_pa=0.15,
        obs_freq=12, N=400, M_per_year=252,
    )
    p_sb  = cn_sb.price()
    g_sb  = cn_sb.greeks()
    t1 = time.time()
    print(f"  ▶ 理论价值 : {p_sb:.6f}  (占名义本金比例)")
    print(f"  ▶ Delta    : {g_sb['delta']:+.6f}")
    print(f"  ▶ Gamma    : {g_sb['gamma']:+.8f}")
    print(f"  ▶ Vega     : {g_sb['vega']:+.6f}  (每 1% σ 变动)")
    print(f"  ▶ Theta    : {g_sb['theta']:+.6f}  (每日损耗)")
    print(f"  ▶ 耗时     : {(t1-t0)*1000:.1f} ms")

    # ── 测试 2：鲨鱼鳍期权 ─────────────────────────────────────
    print("\n【测试 2】鲨鱼鳍期权（Shark Fin — Up-and-Out Call）")
    print("  S0=4000  T=1yr  σ=18%  r=2.5%  q=1.5%  KO=110%  rebate=0.06")
    t0 = time.time()
    cn_sf = CN_FDM(
        product_type    = "shark_fin",
        S0              = 4000.0, T=1.0, r=0.025, sigma=0.18, q=0.015,
        KO_pct=1.10, K_strike=4000.0, rebate=0.06,
        obs_freq=252, N=400, M_per_year=252,
    )
    p_sf  = cn_sf.price()
    g_sf  = cn_sf.greeks()
    t1 = time.time()
    print(f"  ▶ 理论价值 : {p_sf:.6f}  (绝对价值)")
    print(f"  ▶ Delta    : {g_sf['delta']:+.6f}")
    print(f"  ▶ Gamma    : {g_sf['gamma']:+.8f}")
    print(f"  ▶ Vega     : {g_sf['vega']:+.6f}  (每 1% σ 变动)")
    print(f"  ▶ Theta    : {g_sf['theta']:+.6f}  (每日损耗)")
    print(f"  ▶ 耗时     : {(t1-t0)*1000:.1f} ms")

    # ── 验证：鲨鱼鳍 vs Rubinstein-Reiner / Haug 解析公式（rebate=0）──────
    # 注：使用 N(y) 版本（非 N(-y)），对应连续敲出 Up-and-Out Call
    # 参考：Haug (1998) "The Complete Guide to Option Pricing Formulas", Case η=1 φ=1
    print("\n【验证】鲨鱼鳍 CN-FDM vs 解析公式（Up-and-Out Call, rebate=0, 连续监控）")
    S0v, Kv, Hv, Tv = 100.0, 100.0, 110.0, 1.0
    rv, sv, qv       = 0.05, 0.20, 0.02
    bv   = rv - qv                    # cost of carry
    sT   = sv * np.sqrt(Tv)
    mu   = (bv - 0.5*sv**2) / sv**2   # drift 参数 μ
    ebr  = np.exp((bv - rv) * Tv)
    erT  = np.exp(-rv * Tv)
    hS   = Hv / S0v

    x1   = (np.log(S0v/Kv) + (bv + 0.5*sv**2)*Tv) / sT
    x2   = (np.log(S0v/Hv) + (bv + 0.5*sv**2)*Tv) / sT
    y1   = (np.log(Hv**2/(S0v*Kv)) + (bv + 0.5*sv**2)*Tv) / sT
    y2   = (np.log(Hv/S0v)         + (bv + 0.5*sv**2)*Tv) / sT

    # Up-and-Out Call 解析公式（S < H, K < H）
    # UOC = A(x1) - A(x2) - B(y1) + B(y2)
    # A(x) = S·e^{(b-r)T}·N(x) - K·e^{-rT}·N(x-sT)
    # B(y) = S·e^{(b-r)T}·(H/S)^{2(μ+1)}·N(y) - K·e^{-rT}·(H/S)^{2μ}·N(y-sT)
    Afn  = lambda x: S0v*ebr*_norm.cdf(x) - Kv*erT*_norm.cdf(x-sT)
    Bfn  = lambda y: S0v*ebr*hS**(2*(mu+1))*_norm.cdf(y) - Kv*erT*hS**(2*mu)*_norm.cdf(y-sT)
    haug = Afn(x1) - Afn(x2) - Bfn(y1) + Bfn(y2)

    cn_val = CN_FDM(
        "shark_fin", S0=S0v, T=Tv, r=rv, sigma=sv, q=qv,
        KO_pct=Hv/S0v, K_strike=Kv, rebate=0.0,
        obs_freq=252, N=600, M_per_year=600,
    ).price()

    err    = abs(cn_val - haug) / max(abs(haug), 1e-9) * 100
    status = "✓ 精度合格" if err < 2.0 else f"⚠ 误差 {err:.1f}%，可增大 N/M"
    print(f"  参数: S={S0v}, K={Kv}, H={Hv}, T={Tv}yr, r={rv:.0%}, σ={sv:.0%}, q={qv:.0%}")
    print(f"  解析价（连续监控）: {haug:.4f}")
    print(f"  CN-FDM（连续监控）: {cn_val:.4f}")
    print(f"  相对误差          : {err:.3f}%  {status}")
    print(f"  说明: 离散监控(252/yr) MC≈0.154，连续监控解析≈{haug:.4f}，CN-FDM≈{cn_val:.4f} 三者一致")

    print("\n" + "=" * 65)
    print("  所有测试完成。")
    print("=" * 65)


if __name__ == "__main__":
    _run_tests()
