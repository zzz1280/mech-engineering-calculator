"""公差与配合查询（GB/T 1800.1 / 1800.2，公称尺寸 ≤ 500 mm）

【单位 —— 本模块是唯一的例外】
    公称尺寸 : mm
    偏差 / 公差 : μm      ← 公差行业惯例，教材表格也是 μm
    返回时会同时给出 mm 值，方便直接标注到图上。

【为什么这里必须用"查表"，不能用公式】
    ISO 286 确实给了基本偏差的经验公式（例如 d: es = −16·D^0.44），
    但算出来的值与标准表**存在 1~2 μm 的圆整差异**。
    实测：φ30（18~30 段，几何平均 D=√540=23.24 mm）
        公式 d = −16×23.24^0.44 = −63.86 μm，而 GB/T 1800.2 表值 = −65 μm。
    差 1.14 μm —— 对公式推导无所谓，对标在图纸上的公差就是错的。
    所以本模块**内置标准表值**，公式只用于理解，不用于出数。

【数据出处】
    GB/T 1800.2—2020《产品几何技术规范(GPS) 线性尺寸公差ISO代号体系
    第2部分：标准公差带代号和孔、轴的极限偏差表》（修改采用 ISO 286-2）。
    表值经多来源交叉核对（标准原文表、Schaeffler/FAG 轴承样本公差表、
    JIS B 0401 配合表），代码里每个内置数都在 tests/test_fits.py 中有对照用例。

【覆盖范围与已知缺口（诚实声明）】
    · 尺寸段：≤500 mm 的 13 个主段
    · 公差等级：IT5 ~ IT18（IT01~IT4 极少用于配合标注，未内置）
    · 轴基本偏差：c d e f g h js k m n p r s
    · 孔：由同名轴偏差按 GB 规则换算，故同样覆盖 C~H、K~S 等
    · **未内置（会明确报错而不是给错数）**：a、b（基本偏差随公差等级变化）、
      j、t、u、v、x、y、z、za、zb、zc。需要这些代号请查 GB/T 1800.2 原表。
    · 尺寸 > 50 mm 时，r、s 在 GB 中按子段（如 50~65 / 65~80）细分，
      本表取子段首值，返回值里会带 note 提示核对。
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

__all__ = [
    "standard_tolerance",
    "shaft_deviation",
    "hole_deviation",
    "fit",
    "SIZE_SEGMENTS",
]

# --------------------------------------------------------------------------
# 1. 尺寸分段（GB 主段，单位 mm）
# --------------------------------------------------------------------------
SIZE_SEGMENTS: List[Tuple[float, float]] = [
    (0, 3), (3, 6), (6, 10), (10, 18), (18, 30), (30, 50), (50, 80),
    (80, 120), (120, 180), (180, 250), (250, 315), (315, 400), (400, 500),
]


def _segment_index(nominal_mm: float) -> int:
    """返回公称尺寸所属主段下标。φ30 属于 18~30（下标 4）。"""
    if nominal_mm <= 0 or nominal_mm > 500:
        raise ValueError(f"本模块仅支持公称尺寸 (0, 500] mm，当前 {nominal_mm} mm")
    for i, (lo, hi) in enumerate(SIZE_SEGMENTS):
        # 段的左端按"大于"处理：30 属于 18~30 段
        if lo < nominal_mm <= hi:
            return i
    raise ValueError(f"尺寸 {nominal_mm} mm 未落在任何尺寸段内")  # pragma: no cover


# --------------------------------------------------------------------------
# 2. 标准公差 IT 值（μm），IT4 也需要 —— 孔的 Δ 换算要用到 IT(n−1)
# --------------------------------------------------------------------------
_STANDARD_TOLERANCE: Dict[int, List[float]] = {
    4:  [3, 4, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20],
    5:  [4, 5, 6, 8, 9, 11, 13, 15, 18, 20, 23, 25, 27],
    6:  [6, 8, 9, 11, 13, 16, 19, 22, 25, 29, 32, 36, 40],
    7:  [10, 12, 15, 18, 21, 25, 30, 35, 40, 46, 52, 57, 63],
    8:  [14, 18, 22, 27, 33, 39, 46, 54, 63, 72, 81, 89, 97],
    9:  [25, 30, 36, 43, 52, 62, 74, 87, 100, 115, 130, 140, 155],
    10: [40, 48, 58, 70, 84, 100, 120, 140, 160, 185, 210, 230, 250],
    11: [60, 75, 90, 110, 130, 160, 190, 220, 250, 290, 320, 360, 400],
    12: [100, 120, 150, 180, 210, 250, 300, 350, 400, 460, 520, 570, 630],
    13: [140, 180, 220, 270, 330, 390, 460, 540, 630, 720, 810, 890, 970],
    14: [250, 300, 360, 430, 520, 620, 740, 870, 1000, 1150, 1300, 1400, 1550],
    15: [400, 480, 580, 700, 840, 1000, 1200, 1400, 1600, 1850, 2100, 2300, 2500],
    16: [600, 750, 900, 1100, 1300, 1600, 1900, 2200, 2500, 2900, 3200, 3600, 4000],
    17: [1000, 1200, 1500, 1800, 2100, 2500, 3000, 3500, 4000, 4600, 5200, 5700, 6300],
    18: [1400, 1800, 2200, 2700, 3300, 3900, 4600, 5400, 6300, 7200, 8100, 8900, 9700],
}


def standard_tolerance(nominal_mm: float, grade: int) -> float:
    """查标准公差 IT 值。nominal_mm : mm；grade : IT 等级（5~18）；返回 μm。"""
    idx = _segment_index(nominal_mm)
    if grade not in _STANDARD_TOLERANCE:
        raise ValueError(f"暂不支持 IT{grade}，本模块覆盖 IT5~IT18")
    return float(_STANDARD_TOLERANCE[grade][idx])


# --------------------------------------------------------------------------
# 3. 轴的基本偏差（μm）
#    a~h：基本偏差是**上偏差 es**；k~zc：基本偏差是**下偏差 ei**
# --------------------------------------------------------------------------
_SHAFT_UPPER_DEVIATION: Dict[str, List[float]] = {   # es（a~h）
    "c": [-60, -70, -80, -95, -110, -120, -130, -150, -170, -190, -210, -230, -240],
    "d": [-20, -30, -40, -50, -65, -80, -100, -120, -145, -170, -190, -210, -230],
    "e": [-14, -20, -25, -32, -40, -50, -60, -72, -85, -100, -110, -125, -135],
    "f": [-6, -10, -13, -16, -20, -25, -30, -36, -43, -50, -56, -62, -68],
    "g": [-2, -4, -5, -6, -7, -9, -10, -12, -14, -15, -17, -18, -20],
    "h": [0] * 13,
}

# 说明：None 表示该尺寸段**没有可靠数据来源**，查到就报错，绝不拿记忆里的数填。
#   · r：只到 80~120 段（前 8 段），更大尺寸的 GB 子段值未能交叉核实
#   · s：只到 50~80 段（前 7 段），同上
_SHAFT_LOWER_DEVIATION: Dict[str, List[float | None]] = {   # ei（k~s）
    "k": [0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5],
    "m": [2, 4, 6, 7, 8, 9, 11, 13, 15, 17, 20, 21, 23],
    "n": [4, 8, 10, 12, 15, 17, 20, 23, 27, 31, 34, 37, 40],
    "p": [6, 12, 15, 18, 22, 26, 32, 37, 43, 50, 56, 62, 68],
    "r": [10, 15, 19, 23, 28, 34, 41, 51, None, None, None, None, None],
    "s": [14, 19, 23, 28, 35, 43, 53, None, None, None, None, None, None],
}

# 需要提示"GB 按子段细分"的代号：尺寸 > 50 mm 时本表取子段首值
_SUBSEGMENT_SENSITIVE = {"r", "s"}


def shaft_deviation(nominal_mm: float, letter: str, grade: int) -> Dict:
    """查轴的上下偏差。

    nominal_mm : 公称尺寸 (mm)
    letter     : 公差带代号字母，小写，如 'g'、'k6' 里的 'k'
    grade      : IT 等级，如 6

    返回 dict：es / ei（μm 与 mm）、公差值、基本偏差名称。
    计算规则：
      · a~h 与 js：基本偏差 = es，ei = es − IT
      · k~zc：    基本偏差 = ei，es = ei + IT
      · js：      对称公差 ±IT/2
    """
    letter = letter.strip().lower()
    idx = _segment_index(nominal_mm)
    IT = standard_tolerance(nominal_mm, grade)

    note = None
    if nominal_mm > 50 and letter in _SUBSEGMENT_SENSITIVE:
        note = (f"尺寸 >50 mm 时，'{letter}' 在 GB/T 1800.2 中按子段细分，"
                f"本表取子段首值，精确值请核对标准表")

    if letter == "js":
        half = IT / 2.0
        return _pack_deviation("轴", nominal_mm, letter, grade, half, -half, IT, "js 对称公差 ±IT/2", note)

    if letter in _SHAFT_UPPER_DEVIATION:
        es = float(_SHAFT_UPPER_DEVIATION[letter][idx])
        ei = es - IT
        return _pack_deviation("轴", nominal_mm, letter, grade, es, ei, IT,
                               f"基本偏差 es={es:g} μm，ei = es − IT{grade}", note)

    if letter in _SHAFT_LOWER_DEVIATION:
        ei = _SHAFT_LOWER_DEVIATION[letter][idx]
        if ei is None:
            lo, hi = SIZE_SEGMENTS[idx]
            raise ValueError(
                f"轴公差带 '{letter}' 在 {lo}~{hi} mm 尺寸段的数据本项目未内置"
                f"（未能找到可交叉核实的标准值），请查 GB/T 1800.2 原表。"
                f"给不出标准值时，宁可报错也不给近似值。"
            )
        ei = float(ei)
        # k 的特例：IT8 及以上时 k 的基本偏差为 0（GB/T 1800.1）
        if letter == "k" and grade >= 8:
            ei = 0.0
        es = ei + IT
        return _pack_deviation("轴", nominal_mm, letter, grade, es, ei, IT,
                               f"基本偏差 ei={ei:g} μm，es = ei + IT{grade}", note)

    raise ValueError(
        f"轴公差带代号 '{letter}' 暂未内置。已覆盖："
        f"{', '.join(sorted(list(_SHAFT_UPPER_DEVIATION) + list(_SHAFT_LOWER_DEVIATION) + ['js']))}。"
        f"其余代号（a、b、j、t、u、v、x、y、z 等）请查 GB/T 1800.2 原表——"
        f"给不出标准值时，宁可报错也不给近似值。"
    )


# --------------------------------------------------------------------------
# 4. 孔的基本偏差 —— 由同名轴偏差按 GB 规则换算
# --------------------------------------------------------------------------
def hole_deviation(nominal_mm: float, letter: str, grade: int) -> Dict:
    """查孔的上下偏差（ES / EI，μm）。letter 用大写，如 'H'、'K'。

    GB/T 1800.1 换算规则（这部分是本模块最需要你拿教材核对的地方）：
      · A~H：  EI = −es(同名小写轴)，ES = EI + IT          （不加 Δ）
      · J~ZC： ES = −ei(同名小写轴) + Δ，EI = ES − IT
               其中 Δ = IT_n − IT_(n−1)，即本级与高一级公差之差
      · JS：   ±IT/2

    为什么 J~ZC 要加 Δ：为了**保证同名配合互换**——
    "基孔制 H7/k6" 与 "基轴制 K7/h6" 的极限间隙/过盈必须完全相同，
    而纯取相反数算出来的 K7 会让这对配合对不上。加 Δ 后二者精确吻合，
    这一条在 tests/test_fits.py 里有专门的用例验证（φ30 处 K7=+6/−15）。
    """
    raw = letter.strip()
    letter_up = raw.upper()

    if letter_up == "JS":
        IT = standard_tolerance(nominal_mm, grade)
        return _pack_deviation("孔", nominal_mm, "JS", grade, IT / 2.0, -IT / 2.0, IT,
                               "JS 对称公差 ±IT/2")

    lower = letter_up.lower()
    if lower not in set(_SHAFT_UPPER_DEVIATION) | set(_SHAFT_LOWER_DEVIATION):
        raise ValueError(
            f"孔公差带代号 '{raw}' 暂未内置（依赖同名轴偏差换算）。"
            f"已覆盖：{', '.join(sorted(list(_SHAFT_UPPER_DEVIATION) + list(_SHAFT_LOWER_DEVIATION) + ['JS']))}"
        )

    IT = standard_tolerance(nominal_mm, grade)

    if lower in _SHAFT_UPPER_DEVIATION:
        # A~H：EI = −es（通用规则，不加 Δ）
        shaft = shaft_deviation(nominal_mm, lower, grade)
        EI = -shaft["es_um"]
        ES = EI + IT
        note = f"EI = −es({lower}) = {EI:g} μm（A~H 通用规则）；ES = EI + IT{grade}"
    else:
        # J~ZC：ES = −ei + Δ
        if grade - 1 not in _STANDARD_TOLERANCE:
            raise ValueError(f"IT{grade} 无法换算：缺少 IT{grade-1} 数据来计算 Δ")
        delta = IT - standard_tolerance(nominal_mm, grade - 1)
        shaft = shaft_deviation(nominal_mm, lower, grade)
        ES = -shaft["ei_um"] + delta
        EI = ES - IT
        note = (f"ES = −ei({lower}) + Δ = −{shaft['ei_um']:g} + {delta:g} = {ES:g} μm；"
                f"EI = ES − IT{grade}；Δ = IT{grade} − IT{grade-1}")

    return _pack_deviation("孔", nominal_mm, letter_up, grade, ES, EI, IT, note)


# --------------------------------------------------------------------------
# 5. 配合计算
# --------------------------------------------------------------------------
def fit(nominal_mm: float, hole_code: str, shaft_code: str) -> Dict:
    """按配合代号算极限间隙 / 过盈。

    hole_code  : 孔公差带，如 'H7'、'K7'
    shaft_code : 轴公差带，如 'g6'、'k6'

    间隙 / 过盈判据（教材定义）：
        孔最大 D_max = D + ES，孔最小 D_min = D + EI
        轴最大 d_max = D + es，轴最小 d_min = D + ei
        最大间隙 X_max = D_max − d_min = ES − ei
        最小间隙 X_min = D_min − d_max = EI − es
        过盈用负间隙表示：Y_max = −X_min，Y_min = −X_max
    """
    hole_letter, hole_grade = _split_code(hole_code)
    shaft_letter, shaft_grade = _split_code(shaft_code)

    hole = hole_deviation(nominal_mm, hole_letter, hole_grade)
    shaft = shaft_deviation(nominal_mm, shaft_letter, shaft_grade)

    ES, EI = hole["ES_um"] if "ES_um" in hole else hole["es_um"], hole["EI_um"]
    es, ei = shaft["es_um"], shaft["ei_um"]

    X_max = ES - ei        # 最大间隙（正=间隙）
    X_min = EI - es        # 最小间隙（负=过盈）

    if X_min >= 0:
        kind = "间隙配合"
    elif X_max <= 0:
        kind = "过盈配合"
    else:
        kind = "过渡配合"

    base_system = "基孔制（孔基本偏差为 H）" if hole_letter.upper() == "H" \
        else ("基轴制（轴基本偏差为 h）" if shaft_letter.lower() == "h" else "混合制")

    return {
        "nominal_mm": nominal_mm,
        "fit_code": f"{nominal_mm:g} {hole_code}/{shaft_code}",
        "hole": hole,
        "shaft": shaft,
        "hole_limits_mm": [nominal_mm + EI / 1000.0, nominal_mm + ES / 1000.0],
        "shaft_limits_mm": [nominal_mm + ei / 1000.0, nominal_mm + es / 1000.0],
        "max_clearance_um": X_max,
        "min_clearance_um": X_min,
        "max_interference_um": -X_min if X_min < 0 else None,
        "min_interference_um": -X_max if X_max < 0 else None,
        "fit_kind": kind,
        "base_system": base_system,
        "tolerance_zone_hole_um": ES - EI,
        "tolerance_zone_shaft_um": es - ei,
    }


# --------------------------------------------------------------------------
# 6. 内部工具
# --------------------------------------------------------------------------
def _split_code(code: str) -> Tuple[str, int]:
    """把 'H7' 拆成 ('H', 7)；把 'k6' 拆成 ('k', 6)。"""
    s = code.strip()
    i = len(s)
    while i > 0 and s[i - 1].isdigit():
        i -= 1
    if i == 0 or i == len(s):
        raise ValueError(f"公差带代号格式不正确：'{code}'，应形如 H7 / g6 / js7")
    return s[:i], int(s[i:])


def _pack_deviation(kind: str, nominal_mm: float, letter: str, grade: int,
                    upper: float, lower: float, IT: float, note: str,
                    extra_note: str | None = None) -> Dict:
    """统一打包偏差结果，同时给出 μm 与 mm 两套单位。"""
    out = {
        "kind": kind,
        "nominal_mm": nominal_mm,
        "code": f"{letter}{grade}",
        "IT_grade": grade,
        "IT_um": IT,
        "upper_deviation_um": upper,
        "lower_deviation_um": lower,
        "upper_deviation_mm": upper / 1000.0,
        "lower_deviation_mm": lower / 1000.0,
        "note": note,
    }
    # 孔用 ES/EI，轴用 es/ei，两套名字都给，调用方不容易写错
    out["ES_um"], out["EI_um"] = upper, lower
    out["es_um"], out["ei_um"] = upper, lower
    out["ES_mm"], out["EI_mm"] = upper / 1000.0, lower / 1000.0
    out["es_mm"], out["ei_mm"] = upper / 1000.0, lower / 1000.0
    if extra_note:
        out["warning"] = extra_note
    return out
