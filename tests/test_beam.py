"""梁挠度与弯矩 —— 手算对照测试

【怎么写这组用例】
每一条都遵循同一套路：
  1) 选一组"手算友好"的数字（E、I、L 取 10 的整数次幂，心算就能验）；
  2) 先把教材闭式解的结果手算出来写在注释里；
  3) 再断言程序输出等于它。
如果哪天这些断言红了，说明代码被改坏了，或者——更值得警惕——手算和公式理解有出入。

运行：python -m pytest tests/test_beam.py -v
"""

import math

import pytest

from calcs import beam


# --------------------------------------------------------------------------
# 用例 1：简支梁跨中集中载荷
# --------------------------------------------------------------------------
def test_simply_supported_center_point_load():
    """教材经典：简支梁跨中集中力 P。

    手算：L=2 m, P=1000 N, E=2e11 Pa, I=1e-7 m^4
      δ_max = P·L³/(48EI) = 1000×8 / (48 × 2e11 × 1e-7)
            = 8000 / (48 × 2e4) = 8000/960000 = 8.3333e-3 m = 8.333 mm
      M_max = P·L/4 = 1000×2/4 = 500 N·m
      R_左 = R_右 = P/2 = 500 N
    """
    res = beam.simply_supported(L=2.0, E=2e11, I=1e-7, P=1000.0)

    assert res["max_deflection_m"] == pytest.approx(8.3333e-3, rel=1e-4)
    assert res["max_deflection_mm"] == pytest.approx(8.3333, rel=1e-4)
    # 最大挠度应出现在跨中
    assert res["max_deflection_at_x_m"] == pytest.approx(1.0, abs=0.02)
    # 闭式解与数值扫描一致（模块内部已断言，这里再验一次数值相等）
    assert res["max_deflection_closed_form_m"] == pytest.approx(res["max_deflection_m"], rel=1e-6)

    assert res["max_moment_Nm"] == pytest.approx(500.0, rel=1e-6)
    assert res["max_moment_at_x_m"] == pytest.approx(1.0, abs=0.02)
    assert res["reactions"]["left_N"] == pytest.approx(500.0, rel=1e-9)
    assert res["reactions"]["right_N"] == pytest.approx(500.0, rel=1e-9)


# --------------------------------------------------------------------------
# 用例 2：简支梁全跨均布载荷
# --------------------------------------------------------------------------
def test_simply_supported_uniform_load():
    """手算：L=2 m, q=2000 N/m, E=2e11 Pa, I=1e-7 m^4
      δ_max = 5qL⁴/(384EI) = 5×2000×16 / (384 × 2e11 × 1e-7)
            = 160000 / (384 × 2e4) = 160000/7.68e6 = 2.0833e-2 m = 20.833 mm
      M_max = q·L²/8 = 2000×4/8 = 1000 N·m
      R_左 = R_右 = q·L/2 = 2000 N
    """
    res = beam.simply_supported(L=2.0, E=2e11, I=1e-7, q=2000.0)

    assert res["max_deflection_m"] == pytest.approx(2.0833e-2, rel=1e-4)
    assert res["max_deflection_at_x_m"] == pytest.approx(1.0, abs=0.02)
    assert res["max_moment_Nm"] == pytest.approx(1000.0, rel=1e-6)
    assert res["reactions"]["left_N"] == pytest.approx(2000.0, rel=1e-9)


# --------------------------------------------------------------------------
# 用例 3：悬臂梁自由端集中载荷
# --------------------------------------------------------------------------
def test_cantilever_end_point_load():
    """手算：L=1 m, P=500 N, E=2e11 Pa, I=1e-7 m^4
      δ_max = P·L³/(3EI) = 500 / (3 × 2e11 × 1e-7) = 500/(3×2e4) = 8.3333e-3 m
      M_max = P·L = 500 N·m（发生在固定端 x=0）
    """
    res = beam.cantilever(L=1.0, E=2e11, I=1e-7, P=500.0)

    assert res["max_deflection_m"] == pytest.approx(8.3333e-3, rel=1e-4)
    assert res["max_deflection_at_x_m"] == pytest.approx(1.0, abs=1e-6)   # 自由端最大
    assert res["max_moment_abs_Nm"] == pytest.approx(500.0, rel=1e-6)
    assert res["max_moment_at_x_m"] == pytest.approx(0.0, abs=0.01)       # 固定端


# --------------------------------------------------------------------------
# 用例 4：悬臂梁全跨均布载荷
# --------------------------------------------------------------------------
def test_cantilever_uniform_load():
    """手算：L=1 m, q=1000 N/m, E=2e11, I=1e-7
      δ_max = q·L⁴/(8EI) = 1000 / (8 × 2e4) = 6.25e-3 m
      M_max = q·L²/2 = 500 N·m
    """
    res = beam.cantilever(L=1.0, E=2e11, I=1e-7, q=1000.0)

    assert res["max_deflection_m"] == pytest.approx(6.25e-3, rel=1e-4)
    assert res["max_moment_abs_Nm"] == pytest.approx(500.0, rel=1e-6)


# --------------------------------------------------------------------------
# 用例 5：简支梁集中载荷不在跨中 —— 叠加法最容易错的地方
# --------------------------------------------------------------------------
def test_simply_supported_off_center_point_load():
    """手算：L=3 m, P=900 N, 作用点 a=1 m（b=2 m），E=2e11, I=1e-7

      a=1 < b=2，所以公式里取"短侧" b'=min(a,b)=1
      x* = √((L²−b'²)/3) = √((9−1)/3) = √(8/3) = 1.63299 m（这是"从短侧量起"）
      δ_max = P·b'·(L²−b'²)^1.5 / (9√3 · L · EI)
            = 900 × 1 × 8^1.5 / (9×1.7320508 × 3 × 2e11 × 1e-7)
            8^1.5 = 22.6274
            分子 = 900 × 22.6274 = 20364.7
            分母 = 9×1.7320508×3×2e4 = 935307
            δ = 20364.7 / 935307 = 0.021773 m
      最大挠度点从左侧量：因为 a < b（载荷偏左），x* 从右侧量起 → x = L − 1.63299 = 1.36701 m
      M_max = P·a·b/L = 900×1×2/3 = 600 N·m，出现在载荷点 x=1 m
    """
    res = beam.simply_supported(L=3.0, E=2e11, I=1e-7, P=900.0, a=1.0)

    assert res["max_deflection_m"] == pytest.approx(0.021773, rel=1e-3)
    # 载荷偏左 → 最大挠度点应落在跨中偏左（x < 1.5）
    assert res["max_deflection_at_x_m"] == pytest.approx(1.36701, abs=0.03)
    assert res["max_moment_Nm"] == pytest.approx(600.0, rel=1e-6)
    assert res["max_moment_at_x_m"] == pytest.approx(1.0, abs=0.02)
    # 支反力：R_右 = P·a/L = 900/3 = 300 N，R_左 = 600 N
    assert res["reactions"]["left_N"] == pytest.approx(600.0, rel=1e-9)
    assert res["reactions"]["right_N"] == pytest.approx(300.0, rel=1e-9)


# --------------------------------------------------------------------------
# 用例 6：叠加性 —— 两个相同载荷的结果应等于单个载荷的 2 倍（线弹性）
# --------------------------------------------------------------------------
def test_superposition_linearity():
    """线弹性体系必须满足叠加原理，这是检验"公式没抄错"的强力手段：
    两个 500 N 的跨中集中力，结果应精确等于一个 1000 N 的两倍。
    """
    one = beam.simply_supported(L=2.0, E=2e11, I=1e-7, P=500.0)
    two = beam.simply_supported(L=2.0, E=2e11, I=1e-7, P=1000.0)

    assert two["max_deflection_m"] == pytest.approx(2 * one["max_deflection_m"], rel=1e-9)
    assert two["max_moment_Nm"] == pytest.approx(2 * one["max_moment_Nm"], rel=1e-9)

    # 集中 + 均布 也应等于两者单独作用之和
    combo = beam.simply_supported(L=2.0, E=2e11, I=1e-7, P=500.0, q=1000.0)
    only_p = beam.simply_supported(L=2.0, E=2e11, I=1e-7, P=500.0)
    only_q = beam.simply_supported(L=2.0, E=2e11, I=1e-7, q=1000.0)
    assert combo["max_deflection_m"] == pytest.approx(
        only_p["max_deflection_m"] + only_q["max_deflection_m"], rel=1e-3
    )


# --------------------------------------------------------------------------
# 用例 7：截面惯性矩（图的形状别算错）
# --------------------------------------------------------------------------
def test_section_inertia():
    """矩形 40 mm × 80 mm（高 80 在弯曲方向）：
      I = b·h³/12 = 0.04 × 0.08³ / 12 = 0.04 × 5.12e-4 / 12 = 1.7067e-6 m^4
    实心圆 d=50 mm：
      I = π·d⁴/64 = π × 0.05⁴ / 64 = π × 6.25e-6 / 64 = 3.0680e-7 m^4
    """
    assert beam.rectangular_inertia(0.04, 0.08) == pytest.approx(1.70667e-6, rel=1e-4)
    assert beam.solid_circle_inertia(0.05) == pytest.approx(3.0680e-7, rel=1e-4)


# --------------------------------------------------------------------------
# 用例 8：错误输入必须报错，而不是返回一堆 NaN
# --------------------------------------------------------------------------
def test_invalid_input_raises():
    with pytest.raises(ValueError):
        beam.simply_supported(L=-1.0, E=2e11, I=1e-7, P=100.0)
    with pytest.raises(ValueError):
        beam.simply_supported(L=2.0, E=2e11, I=1e-7, P=100.0, a=5.0)  # 载荷落到梁外
    with pytest.raises(ValueError):
        beam.cantilever(L=0.0, E=2e11, I=1e-7, P=100.0)


# --------------------------------------------------------------------------
# 用例 9：悬臂梁载荷不在自由端 —— 教材里的 δ_B = P·a²(3L−a)/(6EI)
# --------------------------------------------------------------------------
def test_cantilever_load_not_at_free_end():
    """手算：L=2 m，载荷作用在 a=1 m 处（距固定端），P=600 N, E=2e11, I=1e-7
      自由端挠度 δ_B = P·a²(3L−a)/(6EI) = 600×1×(6−1)/(6×2e4) = 3000/1.2e5 = 0.025 m
    """
    res = beam.cantilever(L=2.0, E=2e11, I=1e-7, P=600.0, a=1.0)

    # 自由端挠度 = 曲线上 x=L 处的 y
    free_end_y = res["curve"][-1]["y"]
    assert free_end_y == pytest.approx(0.025, rel=1e-9)
    # 最大挠度就是自由端
    assert res["max_deflection_m"] == pytest.approx(0.025, rel=1e-6)
    # 固定端弯矩 = P·a = 600 N·m
    assert res["max_moment_abs_Nm"] == pytest.approx(600.0, rel=1e-6)
