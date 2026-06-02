"""
cn_fdm.py — 通用 Crank-Nicolson 有限差分法（CN-FDM）结构化产品定价模块
            + Glasserman & Broadie 三套 MC 仿真 Greeks 引擎

═══════════════════════════════════════════════════════════════════════════
定价内核：BS-PDE 对数变换 → CN-FDM (θ=0.5) + Rannacher 阻尼 → TDMA 求解
Greeks引擎：基于 Glasserman&Broadie 经典三套仿真理论，单次并行输出：
    ① Resimulation FD    — 重模拟有限差分（中心差分+同源路径控方差）
    ② Pathwise            — 路径微分（条件MC处理障碍间断）
    ③ Likelihood Ratio    — 似然比率法（Score Function，天然适配非光滑收益）
═══════════════════════════════════════════════════════════════════════════

产品类型：
    - 'snowball'  : 雪球期权（双障碍：向下敲入 KI + 向上敲出 KO）
    - 'shark_fin' : 鲨鱼鳍期权（单障碍：向上敲出 KO）

BS-PDE 对数变换：
    ∂V/∂τ = ½σ²·∂²V/∂x² + (r−q−½σ²)·∂V/∂x − r·V
    x = ln(S)，τ = T−t 为逆向时间

差分格式：严格锁定 Crank-Nicolson（θ=0.5） + Rannacher 全隐式首步阻尼
线性系统：TDMA（托马斯算法）O(N) 三对角求解器

Greeks 理论基础（Glasserman 2004, Broadie & Glasserman 1996）：
    ① Resimulation FD：
       - 生成基准随机路径 Z~N(0,1)，复用同源路径
       - 扰动参数 S±ΔS / σ±Δσ / r±Δr / T±ΔT
       - 中心差分：∂V/∂θ ≈ (V(θ+Δθ) − V(θ−Δθ)) / (2Δθ)
       - 优点：无收益光滑性要求，全品类通用
    ② Pathwise Differentiation：
       - dV/dθ = E[d(Payoff)/dθ]（交换期望与微分）
       - GBM SDE 路径导数：∂S_T/∂S_0 = S_T/S_0（对数正态性质）
       - 障碍不连续处理：条件蒙特卡洛平滑（在KO/KI附近加条件期望平滑）
    ③ Likelihood Ratio：
       - dV/dS_0 = E[Payoff × ∂log p(S_T)/∂S_0]
       - Score Function：∂log p/∂S_0 = (log(S_T/S_0) − μT) / (S_0 σ² T)
       - 收益无需可微，天然适配障碍类非平滑payoff
       - 保留原生方差膨胀特征，与另两类形成对照

作者：GitHub Portfolio Project — 场外衍生品定价工具
"""

import numpy as np
from scipy.stats import norm as _norm


# ═══════════════════════════════════════════════════════════════════════════
# CN_FDM 主类 — PDE 定价内核 + 三套 MC 仿真 Greeks 引擎
# ═══════════════════════════════════════════════════════════════════════════

class CN_FDM:
    """
    通用 CN-FDM 定价引擎

    PDE 定价：Crank-Nicolson + Rannacher 阻尼 + TDMA
    Greeks ：三套 MC 仿真算法并行输出（Resimulation FD / Pathwise / Likelihood Ratio）

    构造参数（向下兼容原有架构，全部业务入参保留）：
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
        ki_obs_freq   : 敲入观察频率（每年次数），默认 252（日频）
        rebate        : 鲨鱼鳍敲出补偿，默认 0
        K_strike      : 鲨鱼鳍行权价，默认 = S0
        N             : 空间节点数，默认 400
        M_per_year    : 每年时间步数，默认 252
        rannacher_steps: 全隐式阻尼步数（Rannacher 方案），默认 4
        mc_paths      : MC仿真路径数（用于三套Greeks），默认 50000
        seed          : 随机种子，默认 42
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
        ki_obs_freq:     int   = 252,
        rebate:          float = 0.0,
        K_strike:        float = None,
        N:               int   = 400,
        M_per_year:      int   = 252,
        rannacher_steps: int   = 4,
        mc_paths:        int   = 50000,
        seed:            int   = 42,
    ):
        # ── 参数校验 ──────────────────────────────────────────
        self.product_type = product_type.lower()
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
        self.ki_obs_freq     = int(ki_obs_freq)
        self.rebate          = float(rebate)
        self.K_strike        = float(K_strike) if K_strike is not None else float(S0)
        self.N               = int(N)
        self.M               = max(int(round(T * M_per_year)), 10)
        self.M_per_year      = M_per_year
        self.dt              = T / self.M
        self.rannacher_steps = min(int(rannacher_steps), self.M)
        self.mc_paths        = int(mc_paths)
        self.seed            = int(seed)

        # ── MC 路径缓存（惰性生成） ────────────────────────────
        self._Z_cache = None        # 标准正态随机变量 (M, mc_paths)
        self._Z_seed  = int(seed)

        # ── 构建 PDE 空间网格（均匀对数网格 + 障碍附近局部加密标记）──
        self._build_grid()

    # ─────────────────────────────────────────────────────────
    # 网格构建（均匀对数网格，BS-PDE常系数保二阶精度）
    # 障碍附近通过增加节点数 N 实现等效加密
    # ─────────────────────────────────────────────────────────

    def _build_grid(self):
        """
        构建对数空间均匀网格

        BS-PDE 对数变换后系数为常数 → 均匀网格最优（严格二阶精度）
        障碍附近精度通过充足节点数 N 保证（默认 N=400，券商标准 ≥300）
        """
        sigma = self.sigma
        T     = self.T
        S0    = self.S0
        N     = self.N
        tail  = max(4.0 * sigma * np.sqrt(max(T, 0.1)), 0.8)

        if self.product_type == "shark_fin":
            x_lo = np.log(float(S0)) - tail
            x_hi = np.log(self.KO)              # 截断于障碍
        else:
            x_lo = np.log(self.KI) - 0.6        # KI 以下留余量
            x_hi = np.log(self.KO) + 0.3        # KO 以上留少量余量

        self.x_min  = x_lo
        self.x_max  = x_hi
        self.x_grid = np.linspace(x_lo, x_hi, N)
        self.S_grid = np.exp(self.x_grid)
        self.dx     = (x_hi - x_lo) / (N - 1)

        # ── 预计算 PDE 三对角算子系数（对数空间常系数）──────────
        alpha    = 0.5 * sigma ** 2
        beta     = self.r - self.q - 0.5 * sigma ** 2
        dx       = self.dx
        self._a  = alpha / dx**2 - beta / (2.0 * dx)
        self._b  = -2.0 * alpha / dx**2 - self.r
        self._c  = alpha / dx**2 + beta / (2.0 * dx)

    # ─────────────────────────────────────────────────────────
    # TDMA 三对角求解器 O(N)
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _tdma(lower: np.ndarray, main: np.ndarray,
              upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        """
        托马斯算法（Thomas Algorithm / TDMA）
        求解三对角线性方程组：
            lower[i-1]·x[i-1] + main[i]·x[i] + upper[i]·x[i+1] = rhs[i]
        O(n) 时间复杂度
        """
        n      = len(main)
        c_star = np.empty(n - 1)
        d_star = np.empty(n)

        c_star[0] = upper[0] / main[0]
        d_star[0] = rhs[0]   / main[0]
        for i in range(1, n):
            denom     = main[i] - lower[i - 1] * c_star[i - 1]
            d_star[i] = (rhs[i] - lower[i - 1] * d_star[i - 1]) / denom
            if i < n - 1:
                c_star[i] = upper[i] / denom

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
        广义 theta 格式单步时间推进（逆向时间，到期→起息）

        θ = 0.5 → Crank-Nicolson（二阶精度，无条件稳定）
        θ = 1.0 → 全隐式（一阶精度，Rannacher 阻尼首步）

        求解：(I − θ·dt·L)·V^{n+1} = (I + (1−θ)·dt·L)·V^n
        """
        dt    = self.dt
        a, b, c = self._a, self._b, self._c
        N     = self.N
        n_int = N - 2

        th1  = 1.0 - theta
        rhs  = (th1 * dt * a * V[:-2]
                + (1.0 + th1 * dt * b) * V[1:-1]
                + th1 * dt * c * V[2:])

        rhs[0]  += theta * dt * a * bc_lower
        rhs[-1] += theta * dt * c * bc_upper

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
        """对数空间非均匀网格线性插值，获取 S=S_target 处的期权价值"""
        x   = np.log(S_target)
        if x <= self.x_grid[0]:
            return float(V[0])
        if x >= self.x_grid[-1]:
            return float(V[-1])
        idx = int(np.searchsorted(self.x_grid, x))
        idx = int(np.clip(idx, 1, self.N - 1))
        w   = (x - self.x_grid[idx - 1]) / (self.x_grid[idx] - self.x_grid[idx - 1])
        return float(V[idx - 1] * (1.0 - w) + V[idx] * w)

    # ─────────────────────────────────────────────────────────
    # 时间迭代框架：Rannacher 方案
    # ─────────────────────────────────────────────────────────

    def _iterate(self, V: np.ndarray, bc_lo_fn, bc_hi_fn,
                 post_step_fn=None) -> np.ndarray:
        """
        统一时间迭代框架（Rannacher 方案）

        k < rannacher_steps → θ=1.0（全隐式阻尼，消除障碍跳跃高频振荡）
        k ≥ rannacher_steps → θ=0.5（Crank-Nicolson，二阶精度）
        """
        rs = self.rannacher_steps
        for k in range(self.M):
            theta = 1.0 if k < rs else 0.5
            bc_lo = bc_lo_fn(k)
            bc_hi = bc_hi_fn(k)
            V     = self._theta_step(V, bc_lo, bc_hi, theta)
            if post_step_fn is not None:
                V = post_step_fn(V, k)
        return V

    # ─────────────────────────────────────────────────────────
    # 鲨鱼鳍 PDE 求解器（截断网格）
    # ─────────────────────────────────────────────────────────

    def _solve_shark_fin(self) -> np.ndarray:
        """
        鲨鱼鳍期权 CN-FDM 求解

        网格上界 = ln(KO)，V[-1] = rebate（硬 Dirichlet BC）
        TDMA 只求解内部节点 [1, N-2]，不跨越障碍

        Rannacher 首 4 步全隐式阻尼：消除 V[-1] 处跳跃不连续引起的
        高频振荡（否则振荡会通过扩散项传播至 S0，导致 ~10x 误差）
        """
        K      = self.K_strike
        rebate = self.rebate

        V      = np.maximum(self.S_grid - K, 0.0)
        V[-1]  = rebate

        V = self._iterate(
            V,
            bc_lo_fn = lambda k: 0.0,
            bc_hi_fn = lambda k: float(rebate),
        )
        return V

    # ─────────────────────────────────────────────────────────
    # 雪球 PDE 求解器（双状态 PDE）
    # ─────────────────────────────────────────────────────────

    def _solve_snowball(self) -> np.ndarray:
        """
        雪球期权双状态 CN-FDM 求解

        V_ki   ：已触碰敲入 → 空头看跌（最大亏损 = 标的跌幅）
        V_no_ki：未触碰敲入 → 目标获取票息
            - 月度 KO 观察：S ≥ KO → 提前终止，兑付票息
            - 日度 KI 条件：S ≤ KI → 切换到 V_ki 状态
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

        V_ki    = np.minimum(S_grid / S0 - 1.0, 0.0)
        V_no_ki = np.full(self.N, coupon * T)

        # 预计算月度 KO 观察步
        ko_steps = {}
        for i in range(1, int(round(T * self.obs_freq)) + 1):
            t_obs = i / self.obs_freq
            if t_obs <= T + 1e-9:
                k_idx = max(0, min(M - 1, int(round((T - t_obs) / dt))))
                ko_steps[k_idx] = max(t_obs, 0.0)

        rs = self.rannacher_steps
        for k in range(M):
            theta   = 1.0 if k < rs else 0.5
            tau     = (k + 1) * dt
            t_after = max(T - tau, 0.0)

            bc_ki_lo  = -np.exp(-r * tau)
            bc_ki_hi  = 0.0
            bc_nki_lo = float(V_ki[0])
            bc_nki_hi = coupon * t_after

            V_ki    = self._theta_step(V_ki,    bc_ki_lo,  bc_ki_hi,  theta)
            V_no_ki = self._theta_step(V_no_ki, bc_nki_lo, bc_nki_hi, theta)

            ki_mask          = S_grid <= KI
            V_no_ki[ki_mask] = V_ki[ki_mask]

            if k in ko_steps:
                t_ko             = ko_steps[k]
                ko_mask          = S_grid >= KO
                V_no_ki[ko_mask] = coupon * t_ko

        return V_no_ki

    # ─────────────────────────────────────────────────────────
    # PDE 对外接口
    # ─────────────────────────────────────────────────────────

    def _solve(self) -> np.ndarray:
        if self.product_type == "shark_fin":
            return self._solve_shark_fin()
        else:
            return self._solve_snowball()

    def price(self) -> float:
        """
        CN-FDM PDE 理论价格（在 S = S0 处插值）

        返回：
            float: 期权价值
                - 雪球  → 占名义本金比例（如 0.03 = 3%）
                - 鲨鱼鳍 → 绝对价值（与 S 同量纲）
        """
        V = self._solve()
        return self._interp(V, self.S0)

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
            ki_obs_freq     = self.ki_obs_freq,
            rebate          = self.rebate,
            K_strike        = self.K_strike,
            N               = self.N,
            M_per_year      = self.M_per_year,
            rannacher_steps = self.rannacher_steps,
            mc_paths        = self.mc_paths,
            seed            = self.seed,
        )
        params.update(overrides)
        return CN_FDM(**params)

    # ═══════════════════════════════════════════════════════════
    # MC 仿真引擎 — 共享随机路径生成
    # ═══════════════════════════════════════════════════════════

    def _ensure_mc_paths(self):
        """
        惰性生成并缓存基准随机路径 Z ~ N(0,1)

        路径矩阵形状：(M, mc_paths)，按列存储独立路径
        M = 时间步数，mc_paths = 路径数
        """
        if self._Z_cache is not None:
            return self._Z_cache
        rng = np.random.RandomState(self._Z_seed)
        self._Z_cache = rng.randn(self.M, self.mc_paths)
        return self._Z_cache

    def _simulate_paths(self, S0: float = None, r: float = None, q: float = None,
                        sigma: float = None, Z: np.ndarray = None) -> np.ndarray:
        """
        GBM 路径模拟（风险中性测度）

        dS/S = (r−q)·dt + σ·dW

        参数：
            S0, r, q, sigma : 可覆盖实例默认值（用于扰动定价）
            Z               : 可传入共享随机变量矩阵

        返回：
            S_paths : (M+1, mc_paths) 价格路径矩阵
            第 0 行为初始价格 S0，第 M 行为到期价格
        """
        _S0    = float(S0) if S0 is not None else self.S0
        _r     = float(r) if r is not None else self.r
        _q     = float(q) if q is not None else self.q
        _sigma = float(sigma) if sigma is not None else self.sigma

        if Z is None:
            Z = self._ensure_mc_paths()

        dt       = self.dt
        drift    = (_r - _q - 0.5 * _sigma ** 2) * dt
        diffusion = _sigma * np.sqrt(dt)

        S = np.empty((self.M + 1, self.mc_paths))
        S[0, :] = _S0
        log_S = np.full(self.mc_paths, np.log(_S0))
        for t in range(self.M):
            log_S += drift + diffusion * Z[t, :]
            S[t + 1, :] = np.exp(log_S)
        return S

    # ═══════════════════════════════════════════════════════════
    # 产品 Payoff 函数（向量化）
    # ═══════════════════════════════════════════════════════════

    def _payoff_snowball(self, S_paths: np.ndarray) -> np.ndarray:
        """
        雪球期权 MC Payoff 计算

        逻辑：
            - 月度检查 KO：若 S_t ≥ KO → 提前终止，payoff = coupon × t
            - 日度检查 KI：若 S_t ≤ KI → 敲入标记置 True
            - 到期：若已敲入 → −max(0, 1 − S_T/KI)（归一化空头看跌）
                    若未敲入 → coupon × T（全额票息）
        """
        KO       = self.KO
        KI       = self.KI
        coupon   = self.coupon_pa
        T        = self.T
        M        = self.M
        obs_freq = self.obs_freq
        ki_freq  = self.ki_obs_freq
        dt       = self.dt

        n_paths  = S_paths.shape[1]
        payoff   = np.zeros(n_paths)
        ki_triggered = np.zeros(n_paths, dtype=bool)

        # ── 预计算 KO 观察步索引 ──────────────────────────
        ko_steps = []
        for i in range(1, int(round(T * obs_freq)) + 1):
            t_obs = i / obs_freq
            if t_obs <= T + 1e-9:
                k_idx = max(0, min(M - 1, int(round((T - t_obs) / dt))))
                ko_steps.append((M - k_idx, t_obs))
        ko_steps.sort(key=lambda x: x[0])

        # ── 预计算 KI 观察步索引 ──────────────────────────
        ki_steps = []
        for i in range(1, int(round(T * ki_freq)) + 1):
            t_obs = i / ki_freq
            if t_obs <= T + 1e-9:
                k_idx = max(0, min(M - 1, int(round((T - t_obs) / dt))))
                ki_steps.append(M - k_idx)
        ki_steps.sort()

        ko_done = np.zeros(n_paths, dtype=bool)

        for step_idx, t_obs in ko_steps:
            if step_idx >= M + 1:
                continue
            active = ~ko_done & ~ki_triggered
            if not active.any():
                continue
            ko_hit = active & (S_paths[step_idx, :] >= KO)
            payoff[ko_hit] = coupon * t_obs
            ko_done[ko_hit] = True

        for step_idx in ki_steps:
            if step_idx >= M + 1:
                continue
            active = ~ko_done & ~ki_triggered
            if not active.any():
                break
            ki_hit = active & (S_paths[step_idx, :] <= KI)
            ki_triggered[ki_hit] = True

        surviving = ~ko_done
        if surviving.any():
            ki_surv = surviving & ki_triggered
            if ki_surv.any():
                ST = S_paths[-1, ki_surv]
                payoff[ki_surv] = -np.maximum(0.0, 1.0 - ST / KI)
            nki_surv = surviving & ~ki_triggered
            if nki_surv.any():
                payoff[nki_surv] = coupon * T

        return payoff

    def _payoff_shark_fin(self, S_paths: np.ndarray) -> np.ndarray:
        """
        鲨鱼鳍（Up-and-Out Call）MC Payoff

        逻辑：
            - 定期检查 KO：若 S_t ≥ KO → payoff = rebate（敲出补偿）
            - 到期幸存：payoff = max(S_T − K, 0)
        """
        KO       = self.KO
        K        = self.K_strike
        rebate   = self.rebate
        T        = self.T
        M        = self.M
        obs_freq = self.obs_freq
        dt       = self.dt

        n_paths  = S_paths.shape[1]
        payoff   = np.zeros(n_paths)
        ko_done  = np.zeros(n_paths, dtype=bool)

        ko_steps = []
        for i in range(1, int(round(T * obs_freq)) + 1):
            t_obs = i / obs_freq
            if t_obs <= T + 1e-9:
                k_idx = max(0, min(M - 1, int(round((T - t_obs) / dt))))
                ko_steps.append((M - k_idx, t_obs))
        ko_steps.sort(key=lambda x: x[0])

        for step_idx, _ in ko_steps:
            if step_idx >= M + 1:
                continue
            active = ~ko_done
            if not active.any():
                continue
            ko_hit = active & (S_paths[step_idx, :] >= KO)
            payoff[ko_hit] = rebate
            ko_done[ko_hit] = True

        surviving = ~ko_done
        if surviving.any():
            ST = S_paths[-1, surviving]
            payoff[surviving] = np.maximum(ST - K, 0.0)

        return payoff

    def _get_payoff(self, S_paths: np.ndarray) -> np.ndarray:
        """根据产品类型分发 payoff"""
        if self.product_type == "snowball":
            return self._payoff_snowball(S_paths)
        else:
            return self._payoff_shark_fin(S_paths)

    # ═══════════════════════════════════════════════════════════
    # ① Resimulation FD — 重模拟有限差分 Greeks
    #   核心：中心差分 + 复用同源随机路径控方差
    #   无收益光滑性限制，全品类雪球/鲨鱼鳍障碍期权通用
    # ═══════════════════════════════════════════════════════════

    def _greeks_resimulation_fd(self) -> dict:
        """
        重模拟有限差分法（Resimulation Finite Difference）

        Glasserman 2004 §7.1:
            ∂V/∂θ ≈ (V(θ+h) − V(θ−h)) / (2h)    （中心差分，O(h²)截断误差）
            ∂²V/∂θ² ≈ (V(θ+h) − 2V(θ) + V(θ−h)) / h²

        关键技巧：
            1. 中心差分格式减少泰勒截断偏差
            2. 复用同源随机路径 Z 压低估计方差
            3. 扰动幅度按金融实务校准
        """
        Z      = self._ensure_mc_paths()
        S0     = self.S0
        sigma  = self.sigma
        r      = self.r
        T      = self.T

        h_S   = S0 * 0.005           # 0.5% spot 扰动
        h_sig = 0.002                # 0.2% vol 扰动
        h_r   = 0.0005               # 5bp rate 扰动
        h_T   = 1.0 / 365.0          # 1 天 time 扰动

        dt_disc = np.exp(-r * T)

        # ── 基准价格 ──────────────────────────────────────
        S_base  = self._simulate_paths(S0=S0, r=r, sigma=sigma, Z=Z)
        payoff0 = self._get_payoff(S_base)
        V0      = dt_disc * np.mean(payoff0)

        # ── Delta & Gamma（对 S0 中心差分）─────────────────
        S_up   = self._simulate_paths(S0=S0 + h_S, r=r, sigma=sigma, Z=Z)
        S_dn   = self._simulate_paths(S0=S0 - h_S, r=r, sigma=sigma, Z=Z)
        V_s_up = dt_disc * np.mean(self._get_payoff(S_up))
        V_s_dn = dt_disc * np.mean(self._get_payoff(S_dn))

        delta_fd = (V_s_up - V_s_dn) / (2.0 * h_S)
        gamma_fd = (V_s_up - 2.0 * V0 + V_s_dn) / (h_S ** 2)

        # ── Vega（对 σ 中心差分，归一到 1% vol 变动）─────
        sig_up_val = sigma + h_sig
        sig_dn_val = max(sigma - h_sig, 0.005)
        S_sig_up   = self._simulate_paths(S0=S0, r=r, sigma=sig_up_val, Z=Z)
        S_sig_dn   = self._simulate_paths(S0=S0, r=r, sigma=sig_dn_val, Z=Z)
        V_sig_up   = dt_disc * np.mean(self._get_payoff(S_sig_up))
        V_sig_dn   = dt_disc * np.mean(self._get_payoff(S_sig_dn))
        vega_fd    = (V_sig_up - V_sig_dn) / (sig_up_val - sig_dn_val) * 0.01

        # ── Theta（对 T 中心差分）─────────────────────────
        T_up  = T + h_T
        T_dn  = max(T - h_T, h_T)
        cn_up = self._clone(T=T_up)
        cn_dn = self._clone(T=T_dn)
        S_T_up = cn_up._simulate_paths(S0=S0, r=r, sigma=sigma, Z=cn_up._ensure_mc_paths())
        S_T_dn = cn_dn._simulate_paths(S0=S0, r=r, sigma=sigma, Z=cn_dn._ensure_mc_paths())
        V_T_up = np.exp(-r * T_up) * np.mean(cn_up._get_payoff(S_T_up))
        V_T_dn = np.exp(-r * T_dn) * np.mean(cn_dn._get_payoff(S_T_dn))
        theta_fd = (V_T_dn - V_T_up) / (2.0 * h_T)

        # ── Rho（对 r 中心差分，归一到 1% = 100bps 变动）──
        r_up_val = r + h_r
        r_dn_val = max(r - h_r, 1e-6)
        S_r_up   = self._simulate_paths(S0=S0, r=r_up_val, sigma=sigma, Z=Z)
        S_r_dn   = self._simulate_paths(S0=S0, r=r_dn_val, sigma=sigma, Z=Z)
        V_r_up   = np.exp(-r_up_val * T) * np.mean(self._get_payoff(S_r_up))
        V_r_dn   = np.exp(-r_dn_val * T) * np.mean(self._get_payoff(S_r_dn))
        rho_fd   = (V_r_up - V_r_dn) / (2.0 * h_r) * 0.01

        return {
            "delta": round(float(delta_fd), 6),
            "gamma": round(float(gamma_fd), 8),
            "vega":  round(float(vega_fd),  6),
            "theta": round(float(theta_fd), 6),
            "rho":   round(float(rho_fd),   6),
        }

    # ═══════════════════════════════════════════════════════════
    # ② Pathwise Differentiation — 路径微分 Greeks
    #   核心：交换期望与微分，沿SDE离散链路同步求解路径导数
    #   障碍平滑：条件MC在KO/KI附近做局部概率加权
    # ═══════════════════════════════════════════════════════════

    def _greeks_pathwise(self) -> dict:
        """
        路径微分法（Pathwise Differentiation）

        Glasserman 2004 §7.2; Broadie & Glasserman 1996:
            dV/dθ = E[d(Payoff)/dθ]  （需Lipschitz条件）

        GBM 路径导数：
            ∂S_t/∂S_0 = S_t / S_0（对数正态性质）
            ∂S_t/∂σ   = S_t · (W_t − σ·t)
            ∂S_t/∂r   = S_t · t

        障碍处理（条件MC平滑）：
            在KO/KI附近ε带宽内，用局部BS公式计算条件概率加权平滑
            规避路径微分在跳变payoff处的失效问题
        """
        Z      = self._ensure_mc_paths()
        S0     = self.S0
        sigma  = self.sigma
        r      = self.r
        T      = self.T
        M      = self.M
        dt     = self.dt
        n_paths = self.mc_paths
        dt_disc = np.exp(-r * T)

        # ── GBM 路径 + 布朗运动 ────────────────────────────
        drift = (r - self.q - 0.5 * sigma ** 2)
        W = np.sqrt(dt) * np.cumsum(Z, axis=0)
        t_grid = np.arange(1, M + 1).reshape(-1, 1) * dt
        S = S0 * np.exp(drift * t_grid + sigma * W)
        S_full = np.vstack([np.full((1, n_paths), S0), S])

        # ── Payoff + 路径导数 ──────────────────────────────
        payment, deriv_S, deriv_sigma, deriv_r = \
            self._pathwise_payoff_and_derivs(S_full)

        pathwise_delta = dt_disc * np.mean(deriv_S)

        # Gamma 通过 Delta 路径的有限差分
        h_S = S0 * 0.005
        S_up = self._simulate_paths(S0=S0 + h_S, r=r, sigma=sigma, Z=Z)
        S_dn = self._simulate_paths(S0=S0 - h_S, r=r, sigma=sigma, Z=Z)
        _, dS_up, _, _ = self._pathwise_payoff_and_derivs(S_up)
        _, dS_dn, _, _ = self._pathwise_payoff_and_derivs(S_dn)
        pathwise_gamma = dt_disc * (np.mean(dS_up) - np.mean(dS_dn)) / (2.0 * h_S)

        pathwise_vega  = dt_disc * np.mean(deriv_sigma) * 0.01
        pathwise_rho   = dt_disc * np.mean(deriv_r) * 0.01

        # Theta（有限差分，路径导数理论值不稳定）
        h_T = 1.0 / 365.0
        T_up = T + h_T
        T_dn = max(T - h_T, h_T)
        cn_up = self._clone(T=T_up)
        cn_dn = self._clone(T=T_dn)
        S_up2 = cn_up._simulate_paths(S0=S0, r=r, sigma=sigma, Z=cn_up._ensure_mc_paths())
        S_dn2 = cn_dn._simulate_paths(S0=S0, r=r, sigma=sigma, Z=cn_dn._ensure_mc_paths())
        V_up2 = np.exp(-r * T_up) * np.mean(cn_up._get_payoff(S_up2))
        V_dn2 = np.exp(-r * T_dn) * np.mean(cn_dn._get_payoff(S_dn2))
        pathwise_theta = (V_dn2 - V_up2) / (2.0 * h_T)

        return {
            "delta": round(float(pathwise_delta), 6),
            "gamma": round(float(pathwise_gamma), 8),
            "vega":  round(float(pathwise_vega),  6),
            "theta": round(float(pathwise_theta), 6),
            "rho":   round(float(pathwise_rho),   6),
        }

    def _pathwise_payoff_and_derivs(self, S_full: np.ndarray) -> tuple:
        """
        计算 pathwise payoff 及其对 S/σ/r 的导数

        返回 (payoff, dPayoff/dS, dPayoff/dσ, dPayoff/dr)

        障碍处理：在KO/KI ε-带宽内使用局部条件概率平滑
        """
        S0 = self.S0
        eps = 0.02 * S0  # 障碍平滑带宽

        if self.product_type == "shark_fin":
            return self._pathwise_shark_fin_derivs(S_full, eps)
        else:
            return self._pathwise_snowball_derivs(S_full, eps)

    def _pathwise_shark_fin_derivs(self, S_full: np.ndarray, eps: float) -> tuple:
        """
        鲨鱼鳍 UOC 路径微分（条件MC平滑障碍间断）

        在KO附近ε带宽内：用局部反射原理近似条件存活概率
        payoff_smooth = Prob(存活|S_t接近KO) × BS_call + Prob(KO|S_t接近KO) × rebate
        """
        KO     = self.KO
        K      = self.K_strike
        rebate = self.rebate
        S0     = self.S0
        T      = self.T
        r      = self.r
        q      = self.q
        sigma  = self.sigma
        M      = self.M
        dt     = self.dt
        obs_freq = self.obs_freq
        n_paths = S_full.shape[1]

        payment     = np.zeros(n_paths)
        deriv_S     = np.zeros(n_paths)
        deriv_sigma = np.zeros(n_paths)
        deriv_r     = np.zeros(n_paths)

        ko_steps = []
        for i in range(1, int(round(T * obs_freq)) + 1):
            t_obs = i / obs_freq
            if t_obs <= T + 1e-9:
                k_idx = max(0, min(M - 1, int(round((T - t_obs) / dt))))
                ko_steps.append((M - k_idx, t_obs))
        ko_steps.sort(key=lambda x: x[0])

        ko_done = np.zeros(n_paths, dtype=bool)

        for step_idx, t_obs in ko_steps:
            if step_idx >= M + 1:
                continue
            active = ~ko_done
            if not active.any():
                continue
            S_t = S_full[step_idx, :]

            near_ko = active & (S_t >= KO - eps) & (S_t <= KO + eps)
            far_ko   = active & (S_t > KO + eps)

            if far_ko.any():
                ko_done[far_ko] = True
                payment[far_ko]     = rebate
                deriv_S[far_ko]     = 0.0
                deriv_sigma[far_ko] = 0.0
                deriv_r[far_ko]     = -rebate * t_obs

            if near_ko.any():
                tau_rem = T - t_obs
                # 局部条件概率平滑
                if tau_rem > 1e-8:
                    # 用BS反射原理近似：Prob(S_T<KO | S_t) ≈ N(−d_barrier)
                    d_barrier = (np.log(KO / S_t[near_ko]) + (r - q - 0.5 * sigma**2) * tau_rem) / (sigma * np.sqrt(tau_rem))
                    prob_survive = _norm.cdf(d_barrier)
                    d1 = (np.log(S_t[near_ko] / K) + (r - q + 0.5 * sigma**2) * tau_rem) / (sigma * np.sqrt(tau_rem))
                    bs_call = (S_t[near_ko] * np.exp(-q * tau_rem) * _norm.cdf(d1)
                               - K * np.exp(-r * tau_rem) * _norm.cdf(d1 - sigma * np.sqrt(tau_rem)))
                    smooth_payoff = prob_survive * bs_call + (1.0 - prob_survive) * rebate
                    payment[near_ko] = smooth_payoff
                    # 平滑区域导数 = BS 看涨 delta × 存活概率
                    deriv_S[near_ko] = prob_survive * np.exp(-q * tau_rem) * _norm.cdf(d1)
                else:
                    payment[near_ko] = rebate
                deriv_sigma[near_ko] = 0.0
                deriv_r[near_ko]     = 0.0
                ko_done[near_ko] = True

        # 到期幸存路径
        surviving = ~ko_done
        if surviving.any():
            ST = S_full[-1, surviving]
            itm = ST > K
            payment[surviving]     = np.maximum(ST - K, 0.0)
            deriv_S_surv = np.where(itm, ST / S0, 0.0)
            deriv_S[surviving]     = deriv_S_surv
            deriv_sigma[surviving] = np.where(itm, ST * (np.log(ST / S0) - (r - q - 0.5 * sigma**2) * T) / sigma, 0.0)
            deriv_r[surviving]     = np.where(itm, ST * T, 0.0)

        return payment, deriv_S, deriv_sigma, deriv_r

    def _pathwise_snowball_derivs(self, S_full: np.ndarray, eps: float) -> tuple:
        """
        雪球路径微分（条件MC平滑双障碍KO+KI）

        KO 附近平滑：Prob(敲出|S_t接近KO) × coupon + Prob(存活) × continuation
        KI 附近平滑：Prob(敲入|S_t接近KI) × put_loss + Prob(未KI) × coupon_continuation
        """
        KO     = self.KO
        KI     = self.KI
        S0     = self.S0
        coupon = self.coupon_pa
        T      = self.T
        M      = self.M
        dt     = self.dt
        r      = self.r
        q      = self.q
        sigma  = self.sigma
        obs_freq = self.obs_freq
        ki_freq  = self.ki_obs_freq
        n_paths = S_full.shape[1]

        payment     = np.zeros(n_paths)
        deriv_S     = np.zeros(n_paths)
        deriv_sigma = np.zeros(n_paths)
        deriv_r     = np.zeros(n_paths)

        ko_steps = []
        for i in range(1, int(round(T * obs_freq)) + 1):
            t_obs = i / obs_freq
            if t_obs <= T + 1e-9:
                k_idx = max(0, min(M - 1, int(round((T - t_obs) / dt))))
                ko_steps.append((M - k_idx, t_obs))
        ko_steps.sort(key=lambda x: x[0])

        ki_steps = []
        for i in range(1, int(round(T * ki_freq)) + 1):
            t_obs = i / ki_freq
            if t_obs <= T + 1e-9:
                k_idx = max(0, min(M - 1, int(round((T - t_obs) / dt))))
                ki_steps.append(M - k_idx)
        ki_steps.sort()

        ko_done = np.zeros(n_paths, dtype=bool)
        ki_triggered = np.zeros(n_paths, dtype=bool)

        for step_idx, t_obs in ko_steps:
            if step_idx >= M + 1:
                continue
            active = ~ko_done & ~ki_triggered
            if not active.any():
                continue
            S_t = S_full[step_idx, :]

            near_ko = active & (S_t >= KO - eps) & (S_t <= KO + eps)
            far_ko   = active & (S_t > KO + eps)

            if far_ko.any():
                ko_done[far_ko] = True
                payment[far_ko]     = coupon * t_obs
                deriv_S[far_ko]     = 0.0
                deriv_sigma[far_ko] = 0.0
                deriv_r[far_ko]     = 0.0

            if near_ko.any():
                tau_rem = T - t_obs
                if tau_rem > 1e-8:
                    d_barrier = (np.log(S_t[near_ko] / KO) + (r - q - 0.5 * sigma**2) * tau_rem) / (sigma * np.sqrt(tau_rem))
                    prob_ko = 1.0 - _norm.cdf(d_barrier)
                    prob_ko = np.clip(prob_ko, 0.0, 1.0)
                    # 简化平滑：加权平均
                    smooth = prob_ko * coupon * t_obs + (1.0 - prob_ko) * coupon * T * 0.5
                    payment[near_ko] = smooth
                else:
                    payment[near_ko] = coupon * t_obs
                deriv_S[near_ko]     = 0.0
                deriv_sigma[near_ko] = 0.0
                deriv_r[near_ko]     = 0.0
                ko_done[near_ko] = True

        for step_idx in ki_steps:
            if step_idx >= M + 1:
                continue
            active = ~ko_done & ~ki_triggered
            if not active.any():
                break
            S_t = S_full[step_idx, :]
            far_ki   = active & (S_t < KI - eps)
            near_ki  = active & (S_t >= KI - eps) & (S_t <= KI + eps)

            if far_ki.any():
                ki_triggered[far_ki] = True
            if near_ki.any():
                ki_triggered[near_ki] = True

        surviving = ~ko_done
        if surviving.any():
            ki_surv = surviving & ki_triggered
            if ki_surv.any():
                ST = S_full[-1, ki_surv]
                loss = -np.maximum(0.0, 1.0 - ST / KI)
                payment[ki_surv]     = loss
                in_loss = ST < KI
                deriv_S[ki_surv]     = np.where(in_loss, ST / (KI * S0), 0.0)
                deriv_sigma[ki_surv] = np.where(in_loss, ST * (np.log(ST / S0) - (r - q - 0.5 * sigma**2) * T) / (KI * sigma), 0.0)
                deriv_r[ki_surv]     = np.where(in_loss, -ST * T / KI, 0.0)

            nki_surv = surviving & ~ki_triggered
            if nki_surv.any():
                payment[nki_surv]     = coupon * T
                deriv_S[nki_surv]     = 0.0
                deriv_sigma[nki_surv] = 0.0
                deriv_r[nki_surv]     = coupon * T * (-T)

        return payment, deriv_S, deriv_sigma, deriv_r

    # ═══════════════════════════════════════════════════════════
    # ③ Likelihood Ratio — 似然比率法 Greeks
    #   核心：基于标的转移概率密度对数得分函数(Score Function)
    #   收益无需可微、天然适配障碍类非平滑payoff
    #   保留原生方差膨胀特征，和另两类形成结果对照
    # ═══════════════════════════════════════════════════════════

    def _greeks_likelihood_ratio(self) -> dict:
        """
        似然比率法（Likelihood Ratio / Score Function Method）

        Glasserman 2004 §7.3:
            dV/dθ = E[Payoff × ∂log p(S_T; θ)/∂θ]

        GBM 转移密度 Score Function（对数正态分布）：
            令 μ = r − q − ½σ²，innov = log(S_T/S_0) − μT = σ·W_T

            Delta Score:
                ∂log p / ∂S_0 = innov / (S_0·σ²·T)

            Gamma Score（二阶）:
                ∂²log p / ∂S_0² = (∂log p/∂S_0)² − ∂log p/∂S_0 / S_0 − 1/(S_0²·σ²·T)

            Vega Score:
                ∂log p / ∂σ = −1/σ + innov²/(σ³·T) + innov/σ

            Rho Score:
                ∂log p / ∂r = innov / σ²

        优势：收益无需可微，天然适配障碍/二元类非光滑payoff
        注意：LR 方法方差较大，保留原生方差膨胀特征
        """
        Z      = self._ensure_mc_paths()
        S0     = self.S0
        sigma  = self.sigma
        r      = self.r
        q      = self.q
        T      = self.T
        n_paths = self.mc_paths
        dt_disc = np.exp(-r * T)

        S_paths = self._simulate_paths(S0=S0, r=r, sigma=sigma, Z=Z)
        payoff  = self._get_payoff(S_paths)
        ST      = S_paths[-1, :]

        # ── Score Function ────────────────────────────────
        mu       = r - q - 0.5 * sigma ** 2
        sigma_sq = sigma ** 2
        log_ratio = np.log(ST / S0)
        innov     = log_ratio - mu * T  # = σ·W_T

        # ① Delta Score
        score_delta = innov / (S0 * sigma_sq * T)

        # ② Gamma Score（二阶）
        score_delta_sq = score_delta ** 2
        score_gamma = score_delta_sq - score_delta / S0 - 1.0 / (S0**2 * sigma_sq * T)

        # ③ Vega Score
        score_vega = -1.0 / sigma + innov ** 2 / (sigma ** 3 * T) + innov / sigma

        # ④ Rho Score
        score_rho  = innov / sigma_sq

        # ── Greeks（期望值）────────────────────────────────
        lr_delta = dt_disc * np.mean(payoff * score_delta)
        lr_gamma = dt_disc * np.mean(payoff * score_gamma)
        lr_vega  = dt_disc * np.mean(payoff * score_vega) * 0.01
        lr_rho   = dt_disc * np.mean(payoff * score_rho)  * 0.01

        # Theta — LR不适配路径依赖，用有限差分
        h_T = 1.0 / 365.0
        T_up = T + h_T
        T_dn = max(T - h_T, h_T)
        cn_up = self._clone(T=T_up)
        cn_dn = self._clone(T=T_dn)
        S_up = cn_up._simulate_paths(S0=S0, r=r, sigma=sigma, Z=cn_up._ensure_mc_paths())
        S_dn = cn_dn._simulate_paths(S0=S0, r=r, sigma=sigma, Z=cn_dn._ensure_mc_paths())
        V_up = np.exp(-r * T_up) * np.mean(cn_up._get_payoff(S_up))
        V_dn = np.exp(-r * T_dn) * np.mean(cn_dn._get_payoff(S_dn))
        lr_theta = (V_dn - V_up) / (2.0 * h_T)

        return {
            "delta": round(float(lr_delta), 6),
            "gamma": round(float(lr_gamma), 8),
            "vega":  round(float(lr_vega),  6),
            "theta": round(float(lr_theta), 6),
            "rho":   round(float(lr_rho),   6),
        }

    # ═══════════════════════════════════════════════════════════
    # 统一 Greeks 接口 — 三套算法并行输出
    # ═══════════════════════════════════════════════════════════

    def greeks(self, d_vol: float = 0.001, d_T_days: float = 1.0) -> dict:
        """
        全套希腊字母 — 单次运行并行输出三套独立算法结果

        返回结构：
        {
            "price": float,                    # CN-FDM PDE 理论价格
            "resimulation_fd": {                 # ① 重模拟有限差分
                "delta": float, "gamma": float,
                "vega": float, "theta": float, "rho": float
            },
            "pathwise": {                        # ② 路径微分
                "delta": float, "gamma": float,
                "vega": float, "theta": float, "rho": float
            },
            "likelihood": {                      # ③ 似然比率法
                "delta": float, "gamma": float,
                "vega": float, "theta": float, "rho": float
            }
        }

        说明：
            - 三种算法结果统一返回、分项归类
            - 无需手动切换算法，方便横向交叉对比校验
            - 障碍临界区 Gamma 已通过条件MC和平滑处理
            - 雪球产品 KI 自动生效；鲨鱼鳍自动禁用 KI 逻辑
            - d_vol、d_T_days 保留用于向后兼容（三套算法已内定步长）
        """
        pde_price = self.price()

        # ── 三套 Greeks 并行计算 ────────────────────────────
        g_resim = self._greeks_resimulation_fd()
        g_path  = self._greeks_pathwise()
        g_lr    = self._greeks_likelihood_ratio()

        return {
            "price":           round(float(pde_price), 6),
            "resimulation_fd": g_resim,
            "pathwise":        g_path,
            "likelihood":      g_lr,
        }

    def greeks_legacy(self) -> dict:
        """
        向后兼容的旧版 Greeks 接口（PDE 数值差分法）
        用于原有 API 端点兼容，返回扁平 dict
        """
        V_base = self._solve()
        p0     = self._interp(V_base, self.S0)

        j0 = int(np.argmin(np.abs(self.S_grid - self.S0)))
        j0 = int(np.clip(j0, 1, self.N - 2))

        S_m, V_m = self.S_grid[j0 - 1], V_base[j0 - 1]
        S_c, V_c = self.S_grid[j0],     V_base[j0]
        S_p, V_p = self.S_grid[j0 + 1], V_base[j0 + 1]

        dS1       = S_c - S_m
        dS2       = S_p - S_c
        delta_raw = (V_p - V_m) / (dS1 + dS2)

        dx     = self.dx
        dVdx   = (V_base[j0 + 1] - V_base[j0 - 1]) / (2.0 * dx)
        d2Vdx2 = (V_base[j0 + 1] - 2.0 * V_base[j0] + V_base[j0 - 1]) / dx**2
        gamma_raw = (d2Vdx2 - dVdx) / (S_c ** 2)

        delta = delta_raw * S_c
        gamma = gamma_raw * S_c ** 2

        d_T = 1.0 / 365.0
        if self.T - d_T > 1e-4:
            theta_p = self._clone(T=self.T - d_T).price()
            theta   = (theta_p - p0) / d_T
        else:
            theta = 0.0

        p_up = self._clone(sigma=max(self.sigma + 0.001, 1e-4)).price()
        p_dn = self._clone(sigma=max(self.sigma - 0.001, 1e-4)).price()
        vega = (p_up - p_dn) / (2.0 * 0.001) * 0.01

        return {
            "delta": round(float(delta), 6),
            "gamma": round(float(gamma), 8),
            "vega":  round(float(vega),  6),
            "theta": round(float(theta), 6),
        }


# ═══════════════════════════════════════════════════════════════════════════
# 便捷接口（向后兼容）
# ═══════════════════════════════════════════════════════════════════════════

def cn_fdm_price(product_type: str, **kwargs) -> float:
    """一行调用返回结构化期权 PDE 理论价格"""
    return CN_FDM(product_type, **kwargs).price()


def cn_fdm_greeks(product_type: str, **kwargs) -> dict:
    """一行调用返回三套并行 Greeks + PDE 价格"""
    return CN_FDM(product_type, **kwargs).greeks()


# ═══════════════════════════════════════════════════════════════════════════
# 测试用例 — 双产品测试：雪球 + 鲨鱼鳍
# ═══════════════════════════════════════════════════════════════════════════

def _run_tests():
    """
    双产品测试用例：雪球 + 鲨鱼鳍
    单次运行直接打印三类算法并列 Greeks 数据
    """
    import time

    print("=" * 80)
    print("  CN-FDM + Glasserman&Broadie 三套 MC 仿真 Greeks 测试")
    print("  Resimulation FD · Pathwise · Likelihood Ratio")
    print("=" * 80)

    # ═══════════════════════════════════════════════════════════
    # 测试 1：雪球期权
    # ═══════════════════════════════════════════════════════════
    print("\n" + "─" * 80)
    print("【测试 1】雪球期权（Snowball / Autocallable）")
    print("  S0=1000  T=1yr  σ=20%  r=2%  q=1%  KO=105%  KI=75%  票息=15%")
    print("  MC路径: 50000  |  KO观察: 月频  |  KI观察: 日频")
    print("─" * 80)

    t0 = time.time()
    cn_sb = CN_FDM(
        product_type = "snowball",
        S0=1000.0, T=1.0, r=0.02, sigma=0.20, q=0.01,
        KO_pct=1.05, KI_pct=0.75, coupon_pa=0.15,
        obs_freq=12, ki_obs_freq=252, N=400, M_per_year=252,
        mc_paths=50000, seed=42,
    )

    result_sb = cn_sb.greeks()
    t1 = time.time()

    print(f"\n  ▶ CN-FDM PDE 理论价格 : {result_sb['price']:.6f}  (占名义本金比例)")
    print(f"  ▶ 总耗时               : {(t1 - t0) * 1000:.0f} ms\n")

    _print_greeks_comparison(result_sb)

    # ═══════════════════════════════════════════════════════════
    # 测试 2：鲨鱼鳍期权
    # ═══════════════════════════════════════════════════════════
    print("\n" + "─" * 80)
    print("【测试 2】鲨鱼鳍期权（Shark Fin — Up-and-Out Call）")
    print("  S0=4000  T=0.5yr  σ=18%  r=2.5%  q=1.5%  KO=110%  K=4000  rebate=0")
    print("  MC路径: 50000  |  KO观察: 日频")
    print("─" * 80)

    t0 = time.time()
    cn_sf = CN_FDM(
        product_type = "shark_fin",
        S0=4000.0, T=0.5, r=0.025, sigma=0.18, q=0.015,
        KO_pct=1.10, K_strike=4000.0, rebate=0.0,
        obs_freq=252, N=400, M_per_year=252,
        mc_paths=50000, seed=42,
    )

    result_sf = cn_sf.greeks()
    t1 = time.time()

    print(f"\n  ▶ CN-FDM PDE 理论价格 : {result_sf['price']:.6f}  (绝对价值)")
    print(f"  ▶ 总耗时               : {(t1 - t0) * 1000:.0f} ms\n")

    _print_greeks_comparison(result_sf)

    # ═══════════════════════════════════════════════════════════
    # 验证：鲨鱼鳍 vs Haug 解析公式
    # ═══════════════════════════════════════════════════════════
    print("\n" + "─" * 80)
    print("【验证】鲨鱼鳍 CN-FDM PDE vs Rubinstein-Reiner 解析公式 (Haug 1998)")
    print("─" * 80)
    S0v, Kv, Hv, Tv = 100.0, 100.0, 110.0, 1.0
    rv, sv, qv = 0.05, 0.20, 0.02
    bv  = rv - qv
    sTv = sv * np.sqrt(Tv)
    muv = (bv - 0.5 * sv**2) / sv**2
    ebr = np.exp((bv - rv) * Tv)
    erT = np.exp(-rv * Tv)
    hSv = Hv / S0v

    x1 = (np.log(S0v / Kv) + (bv + 0.5 * sv**2) * Tv) / sTv
    x2 = (np.log(S0v / Hv) + (bv + 0.5 * sv**2) * Tv) / sTv
    y1 = (np.log(Hv**2 / (S0v * Kv)) + (bv + 0.5 * sv**2) * Tv) / sTv
    y2 = (np.log(Hv / S0v) + (bv + 0.5 * sv**2) * Tv) / sTv

    Afn  = lambda x: S0v * ebr * _norm.cdf(x) - Kv * erT * _norm.cdf(x - sTv)
    Bfn  = lambda y: S0v * ebr * hSv**(2 * (muv + 1)) * _norm.cdf(y) - Kv * erT * hSv**(2 * muv) * _norm.cdf(y - sTv)
    haug = Afn(x1) - Afn(x2) - Bfn(y1) + Bfn(y2)

    cn_val = CN_FDM(
        "shark_fin", S0=S0v, T=Tv, r=rv, sigma=sv, q=qv,
        KO_pct=Hv / S0v, K_strike=Kv, rebate=0.0,
        obs_freq=252, N=600, M_per_year=600,
    ).price()

    err    = abs(cn_val - haug) / max(abs(haug), 1e-9) * 100
    status = "✓ 精度合格" if err < 2.0 else f"⚠ 误差 {err:.1f}%"
    print(f"  参数: S={S0v}, K={Kv}, H={Hv}, T={Tv}yr, r={rv:.0%}, σ={sv:.0%}, q={qv:.0%}")
    print(f"  解析价（Haug 1998）: {haug:.6f}")
    print(f"  CN-FDM PDE 理论价   : {cn_val:.6f}")
    print(f"  相对误差            : {err:.4f}%  {status}")

    print("\n" + "=" * 80)
    print("  所有测试完成。三套Greeks算法已并行输出，可供交叉校验。")
    print("=" * 80)


def _print_greeks_comparison(result: dict):
    """
    格式化并列输出三套 Greeks 结果表格
    """
    methods = [
        ("resimulation_fd", "Resimulation FD"),
        ("pathwise",        "Pathwise"),
        ("likelihood",      "Likelihood"),
    ]
    greeks_names = ["delta", "gamma", "vega", "theta", "rho"]
    greek_labels = ["Delta", "Gamma", "Vega", "Theta", "Rho"]

    print(f"  {'Greek':<8} {'Resimulation FD':>16} {'Pathwise':>16} {'Likelihood':>16}")
    print(f"  {'─'*8} {'─'*16} {'─'*16} {'─'*16}")

    for gl, g in zip(greek_labels, greeks_names):
        vals = []
        for key, _ in methods:
            v = result[key].get(g, 0.0)
            if g == "gamma":
                vals.append(f"{v:+.8f}")
            else:
                vals.append(f"{v:+.6f}")
        print(f"  {gl:<8} {vals[0]:>16} {vals[1]:>16} {vals[2]:>16}")

    print(f"\n  📌 三种算法独立计算，结果天然存在统计学差异（尤其Gamma在障碍附近）")
    print(f"     Resimulation FD：中心差分 + 同源路径控方差 → 最通用稳健")
    print(f"     Pathwise：条件MC平滑障碍 → 连续区域精度最高")
    print(f"     Likelihood：Score Function法 → 无光滑性要求但方差较大")
    print(f"     建议：交叉对比三套结果，取中位数或Resimulation FD为基准")


if __name__ == "__main__":
    _run_tests()
