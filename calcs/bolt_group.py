"""螺栓组受力分析（轴向载荷 / 倾覆力矩 / 扭矩）

【单位 —— 内部 SI】
    力 / 载荷        N
    力矩 / 扭矩      N·mm   ← 注意！机械设计教材里螺栓组力矩习惯用 N·mm（因为尺寸用 mm）
    长度 / 距离      mm     ← 同上，螺栓布置尺寸行业惯例就是 mm

    为什么这里不用 N·m：教材例题里螺栓间距是 80 mm、力矩是 1.2×10⁶ N·mm，
    若强行换成 N·m 反而容易和教材对不上。本模块单独声明单位，边界写清楚即可。

【公式出处】
    濮良贵《机械设计》第 9 版 第 5 章「螺纹连接」——螺栓组连接的结构设计与受力分析。
    邱宣怀《机械设计》第 4 版 相关章节同。
    基本假定（教材原话，务必记住，这是公式成立的前提）：
      1) 被连接件为刚体；
      2) 螺栓的变形在弹性范围内；
      3) 接合面受倾覆力矩后仍保持平面（故螺栓受力与其到翻转轴线的距离成正比）。

【三种典型工况】
    A. 轴向载荷通过螺栓组形心      → 各螺栓均分：F_i = F / z
    B. 倾覆力矩 M（绕翻转轴线 O-O）→ F_i = M·L_i / ΣL_j²，L_i = 螺栓轴线到 O-O 的距离
    C. 扭矩 T（铰制孔螺栓，圆周布置）→ F_i = T / (z·r) = 2T / (z·D)
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

__all__ = [
    "axial_uniform",
    "overturning_about_axis",
    "overturning_circular_flange",
    "torque_circular",
    "transverse_friction_preload",
]


# --------------------------------------------------------------------------
# A. 轴向载荷由螺栓组形心承担 —— 各螺栓均分
# --------------------------------------------------------------------------
def axial_uniform(F: float, z: int) -> Dict:
    """轴向载荷 F 通过螺栓组形心时，各螺栓平均分担。

    F : 总轴向载荷 (N)
    z : 螺栓个数
    公式：F_i = F / z
    教材前提：载荷作用线通过螺栓组形心，且各螺栓型号、预紧力相同。
    """
    if z <= 0:
        raise ValueError("螺栓个数 z 必须为正整数")
    if F < 0:
        raise ValueError("本函数只处理拉伸方向为正的轴向载荷，F 请传非负值")

    per_bolt = F / z
    return {
        "case": "轴向载荷均布",
        "F_total_N": F,
        "z": z,
        "loads_N": [per_bolt] * z,
        "max_load_N": per_bolt,
        "formula": "F_i = F / z",
    }


# --------------------------------------------------------------------------
# B. 倾覆力矩 —— 绕翻转轴线 O-O
# --------------------------------------------------------------------------
def overturning_about_axis(
    M: float,
    distances: Sequence[float],
    F_axial: float = 0.0,
    labels: Sequence[str] | None = None,
) -> Dict:
    """螺栓组受倾覆力矩 M（可叠加轴向载荷 F_axial）。

    参数
    ----
    M         : 倾覆力矩 (N·mm)，使接合面趋于"张开"的方向为正
    distances : 每个螺栓轴线到翻转轴线 O-O 的距离 (mm)，列表长度即螺栓数
                例如两排螺栓 4 个：远排 2 个距 200 mm、近排 2 个距 50 mm → [200, 200, 50, 50]
    F_axial   : 叠加的轴向总载荷 (N)，沿螺栓轴线、通过形心；默认 0
    labels    : 螺栓编号，默认 1, 2, 3...

    公式
    ----
    由 M 引起的第 i 个螺栓载荷：  F_i(M) = M · L_i / Σ(L_j²)
    由轴向载荷引起的分量：        F_i(F) = F_axial / z
    第 i 个螺栓总工作载荷：        F_i = F_i(F) + F_i(M)
    最危险螺栓必然是 L_i 最大的那个。
    """
    z = len(distances)
    if z == 0:
        raise ValueError("至少需要一个螺栓")
    if any(d < 0 for d in distances):
        raise ValueError("螺栓到翻转轴线的距离不能为负")
    if M < 0:
        raise ValueError("倾覆力矩 M 请传非负值（方向由 distances 的正负侧体现）")

    sum_L2 = sum(d ** 2 for d in distances)
    if sum_L2 == 0:
        raise ValueError("所有螺栓到翻转轴线的距离都为 0，无法传递倾覆力矩（力矩无着力点）")

    if labels is None:
        labels = [str(i + 1) for i in range(z)]

    loads, details = [], []
    for label, L_i in zip(labels, distances):
        f_m = M * L_i / sum_L2          # 倾覆力矩分量
        f_f = F_axial / z               # 轴向载荷均布分量
        total = f_f + f_m
        loads.append(total)
        details.append({
            "bolt": label,
            "distance_mm": L_i,
            "from_moment_N": f_m,
            "from_axial_N": f_f,
            "total_N": total,
        })

    max_idx = max(range(z), key=lambda i: loads[i])
    return {
        "case": "倾覆力矩（绕翻转轴线）+ 轴向载荷",
        "M_Nmm": M,
        "F_axial_N": F_axial,
        "z": z,
        "sum_distance_squared_mm2": sum_L2,
        "bolts": details,
        "max_load_N": loads[max_idx],
        "max_load_bolt": labels[max_idx],
        "formula": "F_i = F_axial/z + M·L_i / ΣL_j²，最危险螺栓为 L_i 最大者",
    }


def overturning_circular_flange(
    M: float,
    z: int,
    D: float,
    F_axial: float = 0.0,
) -> Dict:
    """圆形凸缘（法兰）上 z 个螺栓均布于直径 D 的圆周，绕一条直径翻转。

    这是工况 B 最常见的具体化：螺栓到翻转轴线的距离 L_i = r·|sin θ_i|，
    均布时可解析求和：ΣL_j² = z·r²/2 （z ≥ 3 均成立，因为 Σsin²θ_i = z/2）

    于是最大螺栓载荷的闭式：
        F_max = M·L_max / (z·r²/2) = 2·M·L_max/(z·r²)

    ⚠️ 一个容易错的点（本项目第一版就踩了）：
    不能想当然写成 F_max = 4M/(z·D)，那等于默认"有螺栓正好落在离轴线最远处 L=r"，
    只有当 z 是 4 的倍数（θ=90° 处有螺栓）时才成立：
        z=4：θ=0,90,180,270 → L_max = r      → 4M/(z·D) 成立
        z=6：θ=0,60,120,...  → L_max = 0.866r → 4M/(z·D) 会**高估 15.5%**
    所以下面用 max(distances) 实测值代入，不再写死 r。

    M : 倾覆力矩 (N·mm)
    z : 螺栓数（均布）
    D : 螺栓中心圆直径 (mm)
    """
    if z < 3:
        raise ValueError("圆周均布螺栓至少需要 3 个，本闭式才成立")
    if D <= 0:
        raise ValueError("螺栓中心圆直径 D 必须为正数")

    r = D / 2.0
    angles = [2.0 * math.pi * i / z for i in range(z)]
    distances = [r * abs(math.sin(theta)) for theta in angles]   # 到翻转轴线的距离
    labels = [f"{i+1}(θ={math.degrees(a):.0f}°)" for i, a in enumerate(angles)]

    result = overturning_about_axis(M, distances, F_axial=F_axial, labels=labels)
    result["case"] = "圆形凸缘均布螺栓 + 倾覆力矩"
    result["pitch_diameter_mm"] = D
    # L_max 取实测最大值（z 不是 4 的倍数时 < r，见 docstring 的告诫）
    L_max = max(distances)
    result["max_distance_mm"] = L_max
    result["closed_form_max_N"] = 2.0 * M * L_max / (z * r ** 2) + F_axial / z
    result["formula"] = (
        "ΣL_j² = z·r²/2（均布圆周解析求和）；"
        "F_max = 2·M·L_max/(z·r²) + F_axial/z，其中 L_max = r·max|sin θ_i|"
    )
    # 闭式解 vs 逐点求和 —— 又一处自检
    if abs(result["closed_form_max_N"] - result["max_load_N"]) > \
            max(abs(result["closed_form_max_N"]) * 1e-9, 1e-9):
        raise AssertionError("内部自检失败：圆形凸缘闭式解与逐点求和不一致")
    return result


# --------------------------------------------------------------------------
# C. 扭矩 —— 铰制孔螺栓（受剪）
# --------------------------------------------------------------------------
def torque_circular(T: float, z: int, D: float) -> Dict:
    """z 个铰制孔螺栓均布于直径 D 的圆周，承受扭矩 T。

    各螺栓受力大小相等、方向垂直于各自半径：
        F = T / (z · r) = 2T / (z · D)
    T : 扭矩 (N·mm)；D : 螺栓中心圆直径 (mm)
    """
    if z <= 0:
        raise ValueError("螺栓个数 z 必须为正整数")
    if D <= 0:
        raise ValueError("螺栓中心圆直径 D 必须为正数")
    if T < 0:
        raise ValueError("扭矩 T 请传非负值")

    per_bolt = 2.0 * T / (z * D)
    return {
        "case": "铰制孔螺栓圆周布置 + 扭矩（受剪）",
        "T_Nmm": T,
        "z": z,
        "pitch_diameter_mm": D,
        "loads_N": [per_bolt] * z,
        "max_load_N": per_bolt,
        "formula": "F = T/(z·r) = 2T/(z·D)",
    }


def transverse_friction_preload(
    F_transverse: float,
    z: int,
    friction_coef: float = 0.15,
    joints: int = 1,
    safety_factor: float = 1.2,
) -> Dict:
    """普通螺栓靠预紧力产生摩擦力承受横向载荷时，所需的**单个螺栓预紧力**。

    公式（教材）：z·f·F_0·m ≥ K_f·F_Σ    →    F_0 ≥ K_f·F_Σ / (f·z·m)
    F_transverse : 横向总载荷 F_Σ (N)
    z            : 螺栓数
    friction_coef: 接合面摩擦系数 f（钢对钢干摩擦常取 0.15，教材表值）
    joints       : 接合面对数 m（单剪取 1，双剪取 2）
    safety_factor: 可靠性系数 K_f（教材取 1.1~1.3）

    注意：这里是"所需预紧力"，不是工作载荷；预紧力会远大于外载荷，
    这正是普通螺栓承受横向载荷的代价，也是为什么重载场合要用铰制孔螺栓或销。
    """
    if z <= 0 or joints <= 0:
        raise ValueError("螺栓数与接合面对数必须为正整数")
    if friction_coef <= 0:
        raise ValueError("摩擦系数必须为正数")

    F0 = safety_factor * F_transverse / (friction_coef * z * joints)
    return {
        "case": "普通螺栓靠摩擦承受横向载荷",
        "F_transverse_N": F_transverse,
        "z": z,
        "friction_coef": friction_coef,
        "joints": joints,
        "safety_factor": safety_factor,
        "required_preload_N": F0,
        "formula": "F_0 ≥ K_f·F_Σ /(f·z·m)",
    }
