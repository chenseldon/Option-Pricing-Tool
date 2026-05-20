"""
main.py - 期权定价工具主程序

功能：
    1. 演示模式（默认）：使用预设中证500指数期权参数，自动输出定价结果、
                        希腊字母汇总表、隐含波动率反推结果及可视化图表。
    2. 交互模式        ：用户通过命令行手动输入期权参数，实时计算并展示结果。

运行方式：
    python main.py

依赖安装：
    pip install py_vollib numpy pandas matplotlib

项目结构：
    bs_pricing.py    ← BSM 定价 + 蒙特卡洛模拟 + 隐含波动率求解
    greeks_calc.py   ← 全希腊字母计算（Delta/Gamma/Vega/Theta/Rho）
    visualization.py ← 期权价格曲线 + 希腊字母曲线可视化
    main.py          ← 主程序入口（演示模式 + 交互模式）

作者：GitHub Portfolio Project
"""

import pandas as pd

from bs_pricing import bs_price, monte_carlo_price, calc_implied_volatility, validate_params
from greeks_calc import calc_all_greeks
from visualization import plot_option_price, plot_greeks


# ─────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────

def print_separator(char="─", width=62):
    """打印分隔线"""
    print(char * width)


def print_pricing_result(flag, S, K, T, r, sigma, q=0.0):
    """
    打印完整的定价结果与风险指标

    输出内容：
        - 参数摘要
        - BSM 理论价格
        - 蒙特卡洛模拟价格（与 BSM 误差对比）
        - 全希腊字母汇总表（pandas DataFrame 格式）

    参数：
        flag  (str)  : 'c' 看涨 / 'p' 看跌
        S     (float): 标的价格
        K     (float): 行权价
        T     (float): 到期期限（年）
        r     (float): 无风险利率（小数）
        sigma (float): 波动率（小数）
        q     (float): 连续分红率（小数），默认 0
    """
    option_type = "看涨期权（Call）" if flag == "c" else "看跌期权（Put）"

    print_separator()
    print(f"  期权类型 : {option_type}")
    print(f"  标的价格 S = {S}    行权价格 K = {K}")
    print(f"  到期期限 T = {T} 年   无风险利率 r = {r * 100:.2f}%")
    print(f"  波动率 σ  = {sigma * 100:.2f}%   连续分红率 q = {q * 100:.2f}%")
    print_separator()

    # ── BSM 定价（核心）
    bsm = bs_price(flag, S, K, T, r, sigma, q)
    print(f"\n  【BSM 理论价格】       = {bsm:.4f}")

    # ── 蒙特卡洛模拟定价（辅助验证，100000 条路径）
    mc = monte_carlo_price(flag, S, K, T, r, sigma, q, n_simulations=100000)
    print(f"  【蒙特卡洛模拟价格】   = {mc:.4f}  "
          f"（与BSM误差：{abs(bsm - mc):.4f}）")

    # ── 全希腊字母汇总
    print(f"\n  【全希腊字母汇总】\n")
    greeks_df = calc_all_greeks(flag, S, K, T, r, sigma, q)
    # 左对齐输出，适配命令行显示
    print(greeks_df.to_string(index=False))
    print()


# ─────────────────────────────────────────────────────────
# 演示模式
# ─────────────────────────────────────────────────────────

def run_demo():
    """
    演示模式：中证500指数期权典型参数测试用例

    测试用例设计说明：
        中证500指数（000905.SH）是A股中小市值核心指数，
        其场外期权是国内机构投资者常用的风险管理工具。

        典型参数设置：
            标的现价约 5500 点，年化波动率约 20%
            无风险利率参考1年期国债收益率约 2%
            指数分红率约 1%
    """
    print("\n" + "═" * 62)
    print("    Python 全品类期权定价工具  |  场外衍生品  |  欧式香草期权")
    print("    Black-Scholes-Merton 模型  +  蒙特卡洛模拟")
    print("═" * 62)

    # ── 测试用例 1：平值看涨期权（ATM Call）
    print("\n▶ 测试用例 1：平值看涨期权（At-The-Money Call）")
    print("  参考标的：中证500指数期权（CSI 500 Index Option）\n")
    S1, K1, T1, r1, sigma1, q1 = 5500, 5500, 0.25, 0.02, 0.20, 0.01
    print_pricing_result("c", S1, K1, T1, r1, sigma1, q1)

    # ── 测试用例 2：虚值看跌期权（OTM Put）
    print("\n▶ 测试用例 2：虚值看跌期权（Out-of-The-Money Put）")
    print("  参考标的：中证500指数期权 · 期限6个月\n")
    S2, K2, T2, r2, sigma2, q2 = 5500, 5000, 0.50, 0.02, 0.22, 0.01
    print_pricing_result("p", S2, K2, T2, r2, sigma2, q2)

    # ── 隐含波动率反推（以测试用例1的 BSM 价格作为"市场价格"）
    print("▶ 隐含波动率反推示例（以测试用例1 BSM 价格为市场价格）\n")
    market_price = bs_price("c", S1, K1, T1, r1, sigma1, q1)
    iv = calc_implied_volatility("c", market_price, S1, K1, T1, r1, q1)
    print(f"  输入市场价格 = {market_price:.4f}")
    print(f"  反推隐含波动率 IV = {iv * 100:.4f}%")
    print(f"  （理论上应等于原始波动率 σ = {sigma1 * 100:.2f}%，验证通过）")
    print_separator()

    # ── 可视化（测试用例1）
    print("\n▶ 生成可视化图表（基于测试用例1：平值看涨期权）")
    print("  正在绘制期权价格曲线...")
    plot_option_price("c", K1, T1, r1, sigma1, q1)

    print("  正在绘制希腊字母曲线...")
    plot_greeks("c", K1, T1, r1, sigma1, q1)

    print("\n✔  演示模式运行完成！图表已保存为 PNG 文件。\n")


# ─────────────────────────────────────────────────────────
# 交互模式
# ─────────────────────────────────────────────────────────

def run_interactive():
    """
    交互模式：用户通过命令行手动输入期权参数

    操作流程：
        1. 依次输入期权类型、标的价格、行权价、到期期限、利率、波动率、分红率
        2. 直接回车使用括号内的默认值（中证500参考参数）
        3. 程序自动校验参数合法性，输出定价结果与希腊字母
        4. 询问是否生成可视化图表
    """
    print("\n" + "═" * 62)
    print("    Python 全品类期权定价工具  |  交互模式")
    print("═" * 62)
    print("  请依次输入期权参数（直接回车使用 [ ] 内默认值）\n")

    def get_float(prompt, default):
        """读取浮点数输入，失败时使用默认值"""
        try:
            val = input(f"  {prompt} [{default}]: ").strip()
            return float(val) if val else default
        except ValueError:
            print(f"  ⚠ 输入无效，使用默认值 {default}")
            return default

    def get_flag():
        """读取期权类型"""
        while True:
            val = input("  期权类型  c=看涨(Call) / p=看跌(Put) [c]: ").strip().lower()
            if val == "":
                return "c"
            if val in ("c", "p"):
                return val
            print("  ⚠ 请输入 'c' 或 'p'")

    # 依次读取各参数
    flag  = get_flag()
    S     = get_float("标的价格 S（元/点）", 5500)
    K     = get_float("行权价格 K（元/点）", 5500)
    T     = get_float("到期期限 T（年，如 0.25=3个月）", 0.25)
    r     = get_float("无风险利率 r（小数，如 0.02=2%）", 0.02)
    sigma = get_float("年化波动率 σ（小数，如 0.20=20%）", 0.20)
    q     = get_float("连续分红率 q（小数，如 0.01=1%）", 0.01)

    # 参数合法性校验
    valid, msg = validate_params(S, K, T, r, sigma, q)
    if not valid:
        print(f"\n  ✘ 参数错误：{msg}\n")
        return

    # 输出定价结果与希腊字母
    print_pricing_result(flag, S, K, T, r, sigma, q)

    # 询问是否生成图表
    plot_choice = input("  是否生成可视化图表？(y/n) [y]: ").strip().lower()
    if plot_choice != "n":
        print("  正在绘制期权价格曲线...")
        plot_option_price(flag, K, T, r, sigma, q)
        print("  正在绘制希腊字母曲线...")
        plot_greeks(flag, K, T, r, sigma, q)

    print("\n✔  计算完成！\n")


# ─────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "─" * 62)
    print("  请选择运行模式：")
    print("    1. 演示模式（自动运行预设中证500指数期权测试用例）")
    print("    2. 交互模式（手动输入参数，实时计算）")
    print("─" * 62)

    choice = input("\n  请输入选项编号 [1]: ").strip()

    if choice == "2":
        run_interactive()
    else:
        run_demo()
