"""滚动轴承寿命计算（默认深沟球轴承）

【单位】
    载荷 C / P / Fr / Fa / C0r : N
    转速 n                     : r/min（寿命公式直接要它，不换算成 rad/s）
    寿命 L10                   : 10⁶ 转（百万转）
    寿命 L10h                  : h

【公式出处】
    濮良贵《机械设计》第 9 版 第 13 章「滚动轴承」；GB/T 6391 / ISO 281。
    基本额定寿命：L10 = (C / P)^ε               单位 10⁶ 转
    换算成小时：  L10h = 10⁶/(60·n) · (C/P)^ε
    指数 ε：球轴承取 3，滚子轴承取 10/3。
    当量动载荷：P = X·F_r + Y·F_a

【X、Y、e 系数表（单列径向接触球轴承 / 深沟球轴承）】
    由相对轴向载荷 F_a/C_0r 查得 e 与 Y，表中未列的值按**线性插值**取
    （GB/T 6391 与 JIS B 1518 均注明"未列值用线性插值"，多份厂商样本一致）。
    判定：F_a/F_r ≤ e → X=1, Y=0（轴向载荷影响可忽略）
          F_a/F_r > e → X=0.56, Y=插值得出

    数据来源：本表与下列公开资料交叉核对一致 ——
    · JIS B 1518 轴承样本「動等価ラジアル荷重」表（NMB-MinebeaMitsumi 技术资料）
    · Shigley《Mechanical Engineering Design》Table 11-1（X1=1, Y1=0; X2=0.56, Y2 见下表）
    · 国内《机械设计》教材表 13-5（部分教材只列 9 档，本表含 12 档，是 9 档的超集）
"""

from __future__ import annotations

from typing import Dict, List, Tuple

__all__ = [
    "equivalent_dynamic_load",
    "basic_rating_life_revolutions",
    "basic_rating_life_hours",
    "deep_groove_ball_life",
]

# (F_a/C_0r, e, Y) —— 单列深沟球轴承，升序排列
_DEEP_GROOVE_FACTORS: List[Tuple[float, float, float]] = [
    (0.014, 0.19, 2.30),
    (0.021, 0.21, 2.15),
    (0.028, 0.22, 1.99),
    (0.042, 0.24, 1.85),
    (0.056, 0.26, 1.71),
    (0.070, 0.27, 1.63),
    (0.084, 0.28, 1.55),
    (0.110, 0.30, 1.45),
    (0.170, 0.34, 1.31),
    (0.280, 0.38, 1.15),
    (0.420, 0.42, 1.04),
    (0.560, 0.44, 1.00),
]

# 寿命指数
EXPONENT_BALL = 3.0        # 球轴承
EXPONENT_ROLLER = 10.0 / 3.0   # 滚子轴承


def _interpolate_factors(ratio: float) -> Tuple[float, float]:
    """按 F_a/C_0r 线性插值取 (e, Y)。超出表范围时取端点值并夹断。"""
    table = _DEEP_GROOVE_FACTORS
    if ratio <= table[0][0]:
        return table[0][1], table[0][2]
    if ratio >= table[-1][0]:
        return table[-1][1], table[-1][2]
    for i in range(len(table) - 1):
        r0, e0, y0 = table[i]
        r1, e1, y1 = table[i + 1]
        if r0 <= ratio <= r1:
            t = (ratio - r0) / (r1 - r0)
            return e0 + t * (e1 - e0), y0 + t * (y1 - y0)
    raise RuntimeError("插值失败：不应到达这里")  # pragma: no cover


def equivalent_dynamic_load(Fr: float, Fa: float, C0r: float) -> Dict:
    """计算深沟球轴承的当量动载荷 P = X·F_r + Y·F_a。

    Fr  : 径向载荷 (N)
    Fa  : 轴向载荷 (N)
    C0r : 基本额定静载荷 (N)，用来定 e 与 Y

    返回 dict：P、X、Y、e、以及两个判定用的比值。
    """
    if Fr < 0 or Fa < 0:
        raise ValueError("载荷不能为负")
    if C0r <= 0:
        raise ValueError("基本额定静载荷 C0r 必须为正数，否则无法查 e、Y 系数")
    if Fr == 0 and Fa == 0:
        raise ValueError("径向与轴向载荷不能同时为 0")

    ratio_static = Fa / C0r if C0r else 0.0      # F_a/C_0r
    e, Y_table = _interpolate_factors(ratio_static)

    # 注意区分两个 Y：
    #   Y_table  —— 从 F_a/C_0r 查表插值得到的系数表值（查出它只是为了备查）
    #   Y        —— 真正代进 P = X·Fr + Y·Fa 的那个值
    # 当 F_a/F_r ≤ e 时，标准规定轴向载荷对寿命影响可忽略，此时取 Y = 0，
    # 但 Y_table 仍然是 1.77 之类。两者混为一谈是新手最常见的错。
    if Fr == 0:
        # 纯轴向载荷：此时 X·Fr = 0，只能靠 Y·Fa
        ratio_load = float("inf")
        X, Y = 0.56, Y_table
    else:
        ratio_load = Fa / Fr                      # F_a/F_r
        if ratio_load <= e:
            X, Y = 1.0, 0.0                       # 轴向影响可忽略
        else:
            X, Y = 0.56, Y_table

    P = X * Fr + Y * Fa
    return {
        "P_N": P,
        "X": X,
        "Y": Y,
        "Y_table": Y_table,
        "e": e,
        "Fa_over_C0r": ratio_static,
        "Fa_over_Fr": ratio_load,
        "axial_dominant": ratio_load > e,
        "formula": "P = X·F_r + Y·F_a（F_a/F_r ≤ e 时 X=1,Y=0；否则 X=0.56,Y 由 F_a/C_0r 插值）",
    }


def basic_rating_life_revolutions(C: float, P: float, exponent: float = EXPONENT_BALL) -> float:
    """基本额定寿命 L10，单位 10⁶ 转：L10 = (C/P)^ε

    C : 基本额定动载荷 (N)
    P : 当量动载荷 (N)
    """
    if C <= 0:
        raise ValueError("基本额定动载荷 C 必须为正数")
    if P <= 0:
        raise ValueError("当量动载荷 P 必须为正数")
    return (C / P) ** exponent


def basic_rating_life_hours(
    C: float,
    P: float,
    n: float,
    exponent: float = EXPONENT_BALL,
    temperature_factor: float = 1.0,
    load_factor: float = 1.0,
) -> float:
    """基本额定寿命 L10h，单位小时：
        L10h = 10⁶/(60·n) · (f_t·C / (f_p·P))^ε

    n                 : 转速 (r/min)，必须 > 0
    temperature_factor: 温度系数 f_t（教材：工作温度 ≤120℃ 取 1.0）
    load_factor       : 载荷系数 f_p（教材：平稳载荷 1.0~1.2，中等冲击 1.2~1.8）

    注：f_t、f_p 是**老版教材 / GB 旧制**的修正方式，ISO 281:2007 起改用
    a_ISO 修正额定寿命。本项目按教材口径实现，README 会注明这一点。
    """
    if n <= 0:
        raise ValueError("转速 n 必须为正数 (r/min)")
    if temperature_factor <= 0 or load_factor <= 0:
        raise ValueError("温度系数与载荷系数必须为正数")

    L10_rev = ((temperature_factor * C) / (load_factor * P)) ** exponent
    return 1e6 / (60.0 * n) * L10_rev


def deep_groove_ball_life(
    Fr: float,
    Fa: float,
    Cr: float,
    Cor: float,
    n: float,
    load_factor: float = 1.0,
    temperature_factor: float = 1.0,
) -> Dict:
    """一步到位：深沟球轴承寿命计算（当量动载荷 → L10 → L10h）。

    Fr  : 径向载荷 (N)
    Fa  : 轴向载荷 (N)
    Cr  : 基本额定动载荷 (N)，由轴承型号查样本
    Cor : 基本额定静载荷 (N)
    n   : 转速 (r/min)

    返回 dict：当量动载荷、X/Y/e、寿命（百万转 / 小时 / 年按每年 300 天 8 小时折算）。
    """
    if Cr <= 0:
        raise ValueError("基本额定动载荷 Cr 必须为正数")

    load = equivalent_dynamic_load(Fr, Fa, Cor)
    P = load["P_N"]

    L10_rev = basic_rating_life_revolutions(Cr, P)
    L10_h = basic_rating_life_hours(
        Cr, P, n,
        temperature_factor=temperature_factor,
        load_factor=load_factor,
    )

    return {
        "bearing_type": "深沟球轴承（单列径向接触球轴承）",
        "Fr_N": Fr, "Fa_N": Fa, "Cr_N": Cr, "Cor_N": Cor, "n_rpm": n,
        "equivalent_load_N": P,
        "X": load["X"], "Y": load["Y"], "Y_table": load["Y_table"], "e": load["e"],
        "Fa_over_C0r": load["Fa_over_C0r"],
        "Fa_over_Fr": load["Fa_over_Fr"],
        "axial_dominant": load["axial_dominant"],
        "L10_million_rev": L10_rev,
        "L10_hours": L10_h,
        "L10_years_300d_8h": L10_h / (300.0 * 8.0),   # 按每年工作 300 天、每天 8 小时折算
        "exponent": EXPONENT_BALL,
        "formula": "P = X·F_r + Y·F_a；L10 = (Cr/P)³ [10⁶转]；L10h = 10⁶/(60n)·(Cr/P)³",
    }
