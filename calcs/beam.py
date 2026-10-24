"""梁的挠度与弯矩（简支梁 / 悬臂梁）

【单位 —— 内部一律 SI，输入输出层再换算】
    L, a, b, x   长度 / 坐标     m
    P            集中载荷        N
    q            均布载荷        N/m
    E            弹性模量        Pa  （钢取 2.06e11）
    I            截面惯性矩      m^4 （矩形 I = b·h^3/12，b 为宽、h 为高，单位 m）
    y            挠度            m，【正号 = 向下】（与载荷同向，看图最直观）
    M            弯矩            N·m，【正号 = 下凸，即梁下侧受拉】

【符号约定为什么要写死】
    材料力学里"挠度向上为正"和"向下为正"两种写法都存在，混用会差一个负号。
    本模块统一取「向下为正」，理由是梁受重力/载荷就是往下沉，正数=下沉量，
    和"最大挠度 ≤ 许用挠度"这种工程判据的直觉一致。

【公式出处】
    刘鸿文《材料力学》第 6 版 第 6 章「弯曲变形」及附录Ⅱ「梁的挠度与转角公式」；
    孙训方《材料力学》附录Ⅱ。下方每个解析式后都标注了教材里的经典形式，
    便于你拿教材逐项核对。

【实现思路：叠加法】
    教材表里每一种工况给一个公式，代码里若照抄会有 10+ 个分支。
    这里改用叠加原理：写出"任意位置集中载荷"和"全跨均布载荷"两个通解后相加，
    一个函数覆盖教材表里多种工况（纯集中 / 纯均布 / 组合 / 载荷不在跨中）。
    代价是必须自己找最大值点，所以下面同时给了「闭式解」与「数值扫描」两套结果，
    二者必须吻合 —— 这本身就是一道自检题。
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

__all__ = [
    "simply_supported",
    "cantilever",
    "rectangular_inertia",
    "solid_circle_inertia",
]


# --------------------------------------------------------------------------
# 0. 小工具：截面惯性矩
# --------------------------------------------------------------------------
def rectangular_inertia(width: float, height: float) -> float:
    """矩形截面惯性矩 I = b·h³/12。

    b 为平行于中性轴的宽度，h 为垂直于中性轴的高度（弯曲方向上的尺寸）。
    width, height : m  →  返回 I : m^4
    """
    if width <= 0 or height <= 0:
        raise ValueError("截面尺寸必须为正数")
    return width * height ** 3 / 12.0


def solid_circle_inertia(diameter: float) -> float:
    """实心圆截面惯性矩 I = π·d⁴/64。diameter : m → I : m^4"""
    if diameter <= 0:
        raise ValueError("直径必须为正数")
    return math.pi * diameter ** 4 / 64.0


# --------------------------------------------------------------------------
# 1. 内部：把一段梁离散成采样点，交给上层求极值 / 画图
# --------------------------------------------------------------------------
def _sampling(L: float, n: int) -> List[float]:
    """生成 0 ~ L 的 n 个等距采样点。n 至少给 2，否则除零。"""
    if n < 2:
        n = 2
    step = L / (n - 1)
    return [i * step for i in range(n)]


def _max_by_scan(points: List[Dict[str, float]], key: str):
    """在采样点里找某个量的最大值（按绝对值比较）及其位置。"""
    best = max(points, key=lambda p: abs(p[key]))
    return best[key], best["x"]


# --------------------------------------------------------------------------
# 2. 简支梁
# --------------------------------------------------------------------------
def simply_supported(
    L: float,
    E: float,
    I: float,
    P: float = 0.0,
    a: Optional[float] = None,
    q: float = 0.0,
    n: int = 201,
) -> Dict:
    """简支梁：任意位置集中载荷 P + 全跨均布载荷 q 的弯矩与挠度。

    参数
    ----
    L : 跨度 (m)
    E : 弹性模量 (Pa)
    I : 截面惯性矩 (m^4)
    P : 集中载荷 (N)，向下为正；不传或传 0 表示没有集中载荷
    a : 集中载荷作用点到左支座的距离 (m)；默认 L/2（跨中）
    q : 全跨均布载荷 (N/m)，向下为正
    n : 挠度曲线采样点数（阶段 3 画图要用）

    返回 dict，含：支反力、最大挠度（闭式 + 数值扫描）、最大弯矩、采样曲线。

    解析式（向下为正）
    ------------------
    支反力：        R_A = P·b/L + q·L/2      （b = L - a）
    弯矩：          M(x) = R_A·x − q·x²/2 − P·⟨x−a⟩
                    ⟨x−a⟩ 为 Macaulay 括号，x<a 时取 0，否则取 x−a
    挠度（0≤x≤a）： y(x) = P·b·x·(L² − b² − x²)/(6·L·E·I) + q·x·(L³ − 2·L·x² + x³)/(24·E·I)
    挠度（a≤x≤L）： 在左式基础上再叠加 P·(x−a)³/(6·E·I)

    教材经典特例（用来验算）
    ------------------------
    · 跨中集中载荷：δ_max = P·L³/(48EI)，M_max = P·L/4
    · 全跨均布载荷：δ_max = 5qL⁴/(384EI)，M_max = q·L²/8
    · 集中载荷不在跨中（a ≥ b 时）：最大挠度点 x* = √((L²−b²)/3)，
      δ_max = P·b·(L²−b²)^{3/2} / (9√3·L·E·I)
    """
    # ---- 参数校验：这些都是"肉眼可见的错输入"，早报错比算出一堆 NaN 强 ----
    if L <= 0:
        raise ValueError("跨度 L 必须为正数 (m)")
    if E <= 0 or I <= 0:
        raise ValueError("弹性模量 E 与惯性矩 I 必须为正数")
    if a is None:
        a = L / 2.0          # 默认跨中
    if not (0.0 <= a <= L):
        raise ValueError(f"集中载荷位置 a 必须落在 [0, L] 内，当前 a={a}, L={L}")
    if P < 0 or q < 0:
        raise ValueError("本函数只处理向下载荷，P、q 请传非负值")

    b = L - a
    EI = E * I

    # ---- 支反力（对左支座取矩得到 R_B，再由竖向平衡得 R_A）----
    #   ΣM_A = 0：R_B·L = P·a + q·L·(L/2)
    R_right = (P * a + q * L * (L / 2.0)) / L
    R_left = P + q * L - R_right

    # ---- 弯矩与挠度：逐点算，逻辑与解析式一一对应 ----
    curve: List[Dict[str, float]] = []
    for x in _sampling(L, n):
        macaulay = (x - a) if x >= a else 0.0          # ⟨x−a⟩
        M = R_left * x - q * x ** 2 / 2.0 - P * macaulay

        # 挠度 = 集中载荷贡献 + 均布载荷贡献（叠加原理）
        y_point = 0.0
        if P != 0.0 and b > 0:
            # 教材形式：y = P·b·x·(L²−b²−x²)/(6·L·EI)，仅在 0≤x≤a 完整；
            # x>a 时补上 P·(x−a)³/(6EI) 这一项（教材分两段给，此处合成一段）
            y_point += P * b * x * (L ** 2 - b ** 2 - x ** 2) / (6.0 * L * EI)
            y_point += P * macaulay ** 3 / (6.0 * EI)
        if q != 0.0:
            # 均布载荷全跨挠度：y = q·x·(L³ − 2·L·x² + x³)/(24·EI)
            y_point += q * x * (L ** 3 - 2.0 * L * x ** 2 + x ** 3) / (24.0 * EI)

        curve.append({"x": x, "M": M, "y": y_point})

    # ---- 最大挠度：数值扫描 ----
    y_num, y_num_x = _max_by_scan(curve, "y")

    # ---- 最大挠度：闭式解（仅对三种经典工况有现成公式）----
    y_formula, y_formula_x = _simply_supported_closed_form(L, E, I, P, a, q)
    if y_formula is not None:
        # 闭式解和数值扫描必须一致，容差取闭式解的 0.5%（采样点离散带来的误差）
        if abs(y_formula - y_num) > max(abs(y_formula) * 5e-3, 1e-12):
            raise AssertionError(
                "内部自检失败：闭式解与数值扫描不一致 "
                f"(闭式 {y_formula:.6g} m, 扫描 {y_num:.6g} m)，请检查公式或采样点数"
            )

    # ---- 最大弯矩：先用剪力零点解析定位，再用扫描兜底 ----
    M_max, M_max_x = _simply_supported_max_moment(L, P, a, q, R_left)
    M_scan, M_scan_x = _max_by_scan(curve, "M")
    if abs(abs(M_scan) - abs(M_max)) > max(abs(M_max) * 5e-3, 1e-9):
        # 解析求根失败时（例如载荷全为 0）退回扫描值
        M_max, M_max_x = M_scan, M_scan_x

    return {
        "support": "简支梁",
        "L": L, "E": E, "I": I, "P": P, "a": a, "q": q,
        "reactions": {"left_N": R_left, "right_N": R_right},
        "max_deflection_m": y_num,
        "max_deflection_mm": y_num * 1e3,
        "max_deflection_at_x_m": y_num_x,
        "max_deflection_closed_form_m": y_formula,
        "max_moment_Nm": M_max,
        "max_moment_at_x_m": M_max_x,
        "max_moment_abs_Nm": abs(M_max),
        "curve": curve,
    }


def _simply_supported_closed_form(L, E, I, P, a, q):
    """简支梁最大挠度的闭式解；只有经典工况能写出简洁公式，否则返回 (None, None)。"""
    EI = E * I
    b = L - a
    # 工况 1：只有集中载荷
    if P > 0 and q == 0:
        if abs(a - b) < 1e-12:                       # 跨中
            return P * L ** 3 / (48.0 * EI), L / 2.0
        # 载荷不跨中：取"离支座较远的一侧"来用公式，保证 x* 落在长段
        long_b = min(a, b)                            # 公式里的 b 取较短那侧
        x_star = math.sqrt((L ** 2 - long_b ** 2) / 3.0)
        delta = P * long_b * (L ** 2 - long_b ** 2) ** 1.5 / (9.0 * math.sqrt(3) * L * EI)
        # 把"从短侧量起"的 x* 换算成从左侧量起的坐标
        x_from_left = x_star if a >= b else L - x_star
        return delta, x_from_left
    # 工况 2：只有均布载荷
    if q > 0 and P == 0:
        return 5.0 * q * L ** 4 / (384.0 * EI), L / 2.0
    return None, None


def _simply_supported_max_moment(L, P, a, q, R_left):
    """由剪力 V(x)=0 定位最大弯矩点。

    剪力：V(x) = R_left − q·x − P·H(x−a)
    两段分别求零点，落在自己区间内的那个就是真实极值点；
    若没有内点解（比如纯集中载荷且剪力不穿越零点），极值出现在集中载荷作用点。
    """
    candidates = []

    if q > 0:
        # 段 1（x < a）：V = R_left − q·x = 0
        x1 = R_left / q
        if 0 <= x1 <= a:
            candidates.append(x1)
        # 段 2（x ≥ a）：V = R_left − q·x − P = 0
        x2 = (R_left - P) / q
        if a <= x2 <= L:
            candidates.append(x2)
    elif P > 0:
        # 纯集中载荷：弯矩在载荷作用点取极大（剪力在此突变）
        candidates.append(a)
    else:
        return 0.0, 0.0

    best_x, best_M = None, None
    for x in candidates:
        macaulay = (x - a) if x >= a else 0.0
        M = R_left * x - q * x ** 2 / 2.0 - P * macaulay
        if best_M is None or abs(M) > abs(best_M):
            best_x, best_M = x, M
    if best_M is None:
        return 0.0, 0.0
    return best_M, best_x


# --------------------------------------------------------------------------
# 3. 悬臂梁
# --------------------------------------------------------------------------
def cantilever(
    L: float,
    E: float,
    I: float,
    P: float = 0.0,
    a: Optional[float] = None,
    q: float = 0.0,
    n: int = 201,
) -> Dict:
    """悬臂梁：固定端在 x=0，自由端在 x=L。

    参数含义同 simply_supported，但 a 是「集中载荷到固定端的距离」，默认 L（自由端）。

    解析式（向下为正）
    ------------------
    弯矩（x ≤ a）： M(x) = −[ P·(a − x) + q·(L − x)²/2 ]
    弯矩（x > a）：  M(x) = −[ q·(L − x)²/2 ]      （集中载荷在截面左侧，不产生弯矩）
    挠度（x ≤ a）： y = P·x²·(3a − x)/(6EI) + q·x²·(6L² − 4Lx + x²)/(24EI)
    挠度（x > a）： y = P·a²·(3x − a)/(6EI) + q·x²·(6L² − 4Lx + x²)/(24EI)

    教材经典特例
    ------------
    · 自由端集中载荷：δ_max = P·L³/(3EI)，M_max = P·L   （均在固定端）
    · 全跨均布载荷：  δ_max = q·L⁴/(8EI)， M_max = q·L²/2
    · 集中载荷距固定端 a：自由端挠度 δ_B = P·a²·(3L − a)/(6EI)
    """
    if L <= 0:
        raise ValueError("悬臂长度 L 必须为正数 (m)")
    if E <= 0 or I <= 0:
        raise ValueError("弹性模量 E 与惯性矩 I 必须为正数")
    if a is None:
        a = L                       # 默认打在自由端
    if not (0.0 <= a <= L):
        raise ValueError(f"集中载荷位置 a 必须落在 [0, L] 内，当前 a={a}, L={L}")
    if P < 0 or q < 0:
        raise ValueError("本函数只处理向下载荷，P、q 请传非负值")

    EI = E * I
    curve: List[Dict[str, float]] = []
    for x in _sampling(L, n):
        # 弯矩：取截面右侧部分对截面形心取矩，悬臂梁上侧受拉故取负号
        M = -(q * (L - x) ** 2 / 2.0)
        if x <= a:
            M -= P * (a - x)

        # 挠度：集中载荷两段式 + 均布载荷全跨式
        y_point = q * x ** 2 * (6.0 * L ** 2 - 4.0 * L * x + x ** 2) / (24.0 * EI)
        if x <= a:
            y_point += P * x ** 2 * (3.0 * a - x) / (6.0 * EI)
        else:
            y_point += P * a ** 2 * (3.0 * x - a) / (6.0 * EI)

        curve.append({"x": x, "M": M, "y": y_point})

    y_num, y_num_x = _max_by_scan(curve, "y")
    M_num, M_num_x = _max_by_scan(curve, "M")

    # 闭式解自检
    y_formula = None
    if P > 0 and q == 0:
        y_formula = P * L ** 3 / (3.0 * EI) if abs(a - L) < 1e-12 \
            else P * a ** 2 * (3.0 * L - a) / (6.0 * EI)
    elif q > 0 and P == 0:
        y_formula = q * L ** 4 / (8.0 * EI)
    if y_formula is not None:
        if abs(y_formula - y_num) > max(abs(y_formula) * 5e-3, 1e-12):
            raise AssertionError(
                "内部自检失败：悬臂梁闭式解与数值扫描不一致 "
                f"(闭式 {y_formula:.6g} m, 扫描 {y_num:.6g} m)"
            )

    return {
        "support": "悬臂梁",
        "L": L, "E": E, "I": I, "P": P, "a": a, "q": q,
        "reactions": {"fixed_end_N": P + q * L, "fixed_end_moment_Nm": abs(M_num)},
        "max_deflection_m": y_num,
        "max_deflection_mm": y_num * 1e3,
        "max_deflection_at_x_m": y_num_x,
        "max_deflection_closed_form_m": y_formula,
        "max_moment_Nm": M_num,
        "max_moment_at_x_m": M_num_x,
        "max_moment_abs_Nm": abs(M_num),
        "curve": curve,
    }
