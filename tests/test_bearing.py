"""深沟球轴承寿命 —— 手算对照测试

【这组用例要盯的概念】
1) L10 的含义：一批相同轴承中 90% 在出现疲劳点蚀前能达到的寿命，不是"平均寿命"。
2) 指数 ε：球轴承 3、滚子轴承 10/3 —— 载荷减半，球轴承寿命变 8 倍。
   这个"三次方"关系是本计算器最值得讲给用户的直觉。
3) X/Y/e 的判定顺序：先由 F_a/C_0r 定 e 和 Y，再用 F_a/F_r 与 e 比较决定用不用 Y。
   顺序反了结果就错，本文件用例 4 专门覆盖 e 的插值。
"""

import pytest

from calcs import bearing


# --------------------------------------------------------------------------
# 用例 1：寿命公式本体（教材例题口径）
# --------------------------------------------------------------------------
def test_basic_rating_life_hours():
    """手算：C = 20000 N，P = 4000 N，n = 1000 r/min
      (C/P)³ = 5³ = 125
      L10h = 10⁶/(60×1000) × 125 = 16.6667 × 125 = 2083.33 h
      L10  = 125 × 10⁶ 转
    """
    L10h = bearing.basic_rating_life_hours(C=20000.0, P=4000.0, n=1000.0)
    L10 = bearing.basic_rating_life_revolutions(C=20000.0, P=4000.0)

    assert L10 == pytest.approx(125.0)          # 单位：10⁶ 转
    assert L10h == pytest.approx(2083.333, rel=1e-5)


# --------------------------------------------------------------------------
# 用例 2：寿命与载荷的三次方关系
# --------------------------------------------------------------------------
def test_life_cube_law():
    """载荷减半 → 寿命变 8 倍（球轴承 ε=3）。这条最能体现"指数"的物理含义。"""
    base = bearing.basic_rating_life_hours(C=20000.0, P=4000.0, n=1000.0)
    half = bearing.basic_rating_life_hours(C=20000.0, P=2000.0, n=1000.0)

    assert half == pytest.approx(8.0 * base, rel=1e-9)


# --------------------------------------------------------------------------
# 用例 3：纯径向载荷 —— X=1, Y=0
# --------------------------------------------------------------------------
def test_equivalent_load_pure_radial():
    """手算：Fr = 5000 N, Fa = 1000 N, C0r = 20000 N
      F_a/C_0r = 0.05 → 在 0.042(e=0.24,Y=1.85) 与 0.056(e=0.26,Y=1.71) 之间插值
          t = (0.05−0.042)/(0.056−0.042) = 0.008/0.014 = 0.5714
          e = 0.24 + 0.5714×0.02 = 0.25143
          Y = 1.85 + 0.5714×(−0.14) = 1.77
      F_a/F_r = 1000/5000 = 0.2 ≤ e → 实际取 X=1、Y=0
      P = 1×5000 + 0×1000 = 5000 N（轴向载荷影响可忽略）

    注意：查表插值得到了 Y_table = 1.77，但因为判定为"轴向不主导"，
    真正代入公式的是 Y = 0。两个值都在返回值里，别看混了。
    """
    res = bearing.equivalent_dynamic_load(Fr=5000.0, Fa=1000.0, C0r=20000.0)

    assert res["e"] == pytest.approx(0.25143, abs=1e-5)
    assert res["Y_table"] == pytest.approx(1.77, abs=1e-4)   # 查表得到的值
    assert res["Y"] == pytest.approx(0.0)                    # 实际代入公式的值
    assert res["X"] == pytest.approx(1.0)
    assert res["axial_dominant"] is False
    assert res["P_N"] == pytest.approx(5000.0)


# --------------------------------------------------------------------------
# 用例 4：轴向载荷占主导 —— X=0.56, Y 由插值得到
# --------------------------------------------------------------------------
def test_equivalent_load_axial_dominant():
    """手算：Fr = 2000 N, Fa = 1000 N, C0r = 10000 N
      F_a/C_0r = 0.10 → 在 0.084(e=0.28,Y=1.55) 与 0.110(e=0.30,Y=1.45) 之间插值
          t = (0.10−0.084)/(0.110−0.084) = 0.016/0.026 = 0.61538
          e = 0.28 + 0.61538×0.02 = 0.29231
          Y = 1.55 + 0.61538×(−0.10) = 1.48846
      F_a/F_r = 1000/2000 = 0.5 > e → X=0.56, Y=1.48846
      P = 0.56×2000 + 1.48846×1000 = 1120 + 1488.46 = 2608.46 N
    """
    res = bearing.equivalent_dynamic_load(Fr=2000.0, Fa=1000.0, C0r=10000.0)

    assert res["e"] == pytest.approx(0.29231, abs=1e-5)
    assert res["X"] == pytest.approx(0.56)
    assert res["Y"] == pytest.approx(1.48846, abs=1e-5)
    assert res["axial_dominant"] is True
    assert res["P_N"] == pytest.approx(2608.462, abs=1e-3)


# --------------------------------------------------------------------------
# 用例 5：表格端点精确命中（不插值）
# --------------------------------------------------------------------------
def test_table_endpoint_exact():
    """F_a/C_0r = 0.014 应精确取到 e=0.19、Y_table=2.30（表首行，不插值）。

    此处 F_a/F_r = 140/10000 = 0.014 ≤ e，故实际 Y = 0，但查表值仍是 2.30。
    """
    res = bearing.equivalent_dynamic_load(Fr=10000.0, Fa=140.0, C0r=10000.0)

    assert res["Fa_over_C0r"] == pytest.approx(0.014, rel=1e-9)
    assert res["e"] == pytest.approx(0.19)
    assert res["Y_table"] == pytest.approx(2.30)


# --------------------------------------------------------------------------
# 用例 6：一步到位接口（含载荷系数）
# --------------------------------------------------------------------------
def test_deep_groove_ball_life_integrated():
    """手算：Fr = 4000 N, Fa = 0, Cr = 20000 N, Cor = 15000 N, n = 1000 r/min
      F_a/C_0r = 0 → 取表首行 e = 0.19
      F_a/F_r = 0 ≤ 0.19 → X=1, Y=0 → P = 4000 N
      L10  = (20000/4000)³ = 125 ×10⁶ 转
      L10h = 2083.33 h
      若取载荷系数 f_p = 1.5（中等冲击）：
      L10h = 2083.33 / 1.5³ = 2083.33/3.375 = 617.28 h
    """
    res = bearing.deep_groove_ball_life(
        Fr=4000.0, Fa=0.0, Cr=20000.0, Cor=15000.0, n=1000.0
    )
    assert res["equivalent_load_N"] == pytest.approx(4000.0)
    assert res["L10_million_rev"] == pytest.approx(125.0)
    assert res["L10_hours"] == pytest.approx(2083.333, rel=1e-5)

    with_shock = bearing.deep_groove_ball_life(
        Fr=4000.0, Fa=0.0, Cr=20000.0, Cor=15000.0, n=1000.0, load_factor=1.5
    )
    assert with_shock["L10_hours"] == pytest.approx(617.284, rel=1e-5)


# --------------------------------------------------------------------------
# 用例 7：错误输入
# --------------------------------------------------------------------------
def test_invalid_input_raises():
    with pytest.raises(ValueError):
        bearing.basic_rating_life_hours(C=20000.0, P=4000.0, n=0)   # 转速为 0 无法折算小时
    with pytest.raises(ValueError):
        bearing.equivalent_dynamic_load(Fr=0.0, Fa=0.0, C0r=10000.0)  # 完全没载荷
    with pytest.raises(ValueError):
        bearing.equivalent_dynamic_load(Fr=1000.0, Fa=500.0, C0r=0.0)  # 缺 C0r 查不了表
