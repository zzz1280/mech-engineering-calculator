"""公差配合 —— 手算 / 查表对照测试

【这组用例的意义】
公差表的数值**不能靠公式推**，只能查标准。下面每条断言的期望值都直接来自
GB/T 1800.2 表（并与 Schaeffler、FAG 轴承样本、JIS B 0401 配合表交叉核对过）。
如果哪天有人"优化"代码改用公式算，这组测试会立刻变红 —— 这是防回归的护栏。

【核对锚点：φ30（属于 18~30 段）】
IT5=9  IT6=13  IT7=21  IT8=33  IT9=52  IT10=84  IT11=130   （单位 μm）
"""

import pytest

from calcs import fits


# --------------------------------------------------------------------------
# 用例 1：标准公差 IT 值（φ30）
# --------------------------------------------------------------------------
def test_standard_tolerance_phi30():
    """φ30 各等级的标准公差，来自 GB/T 1800.2 表 1。"""
    assert fits.standard_tolerance(30.0, 5) == pytest.approx(9.0)
    assert fits.standard_tolerance(30.0, 6) == pytest.approx(13.0)
    assert fits.standard_tolerance(30.0, 7) == pytest.approx(21.0)
    assert fits.standard_tolerance(30.0, 8) == pytest.approx(33.0)
    assert fits.standard_tolerance(30.0, 9) == pytest.approx(52.0)
    assert fits.standard_tolerance(30.0, 10) == pytest.approx(84.0)
    assert fits.standard_tolerance(30.0, 11) == pytest.approx(130.0)


def test_standard_tolerance_other_segments():
    """换两个尺寸段抽查，确认没有整列错位。
    φ10（6~10 段）：IT6=9, IT7=15, IT11=90
    φ80（50~80 段）：IT6=19, IT7=30, IT11=190
    """
    assert fits.standard_tolerance(10.0, 6) == pytest.approx(9.0)
    assert fits.standard_tolerance(10.0, 7) == pytest.approx(15.0)
    assert fits.standard_tolerance(10.0, 11) == pytest.approx(90.0)

    assert fits.standard_tolerance(80.0, 6) == pytest.approx(19.0)
    assert fits.standard_tolerance(80.0, 7) == pytest.approx(30.0)
    assert fits.standard_tolerance(80.0, 11) == pytest.approx(190.0)


# --------------------------------------------------------------------------
# 用例 2：轴的上下偏差（φ30，间隙配合系列）
# --------------------------------------------------------------------------
@pytest.mark.parametrize("letter,grade,es,ei", [
    ("g", 6, -7.0, -20.0),     # φ30g6
    ("h", 6, 0.0, -13.0),      # φ30h6
    ("h", 7, 0.0, -21.0),      # φ30h7
    ("f", 7, -20.0, -41.0),    # φ30f7
    ("d", 9, -65.0, -117.0),   # φ30d9
    ("c", 11, -110.0, -240.0), # φ30c11
])
def test_shaft_clearance_fits(letter, grade, es, ei):
    """间隙配合常用轴公差带，期望值来自 GB/T 1800.2 轴的极限偏差表。"""
    res = fits.shaft_deviation(30.0, letter, grade)
    assert res["es_um"] == pytest.approx(es)
    assert res["ei_um"] == pytest.approx(ei)
    # 公差带宽度必须等于 IT 值 —— 这条能抓出"上偏差对了、下偏差算错"的情况
    assert res["es_um"] - res["ei_um"] == pytest.approx(fits.standard_tolerance(30.0, grade))


# --------------------------------------------------------------------------
# 用例 3：轴的上下偏差（φ30，过渡 / 过盈配合系列）
# --------------------------------------------------------------------------
@pytest.mark.parametrize("letter,grade,es,ei", [
    ("k", 6, 15.0, 2.0),       # φ30k6
    ("m", 6, 21.0, 8.0),       # φ30m6
    ("n", 6, 28.0, 15.0),      # φ30n6
    ("p", 6, 35.0, 22.0),      # φ30p6
    ("r", 6, 41.0, 28.0),      # φ30r6
    ("s", 6, 48.0, 35.0),      # φ30s6
])
def test_shaft_transition_fits(letter, grade, es, ei):
    """过渡 / 过盈配合常用轴公差带。注意这些代号的基本偏差是**下偏差 ei**。"""
    res = fits.shaft_deviation(30.0, letter, grade)
    assert res["es_um"] == pytest.approx(es)
    assert res["ei_um"] == pytest.approx(ei)


# --------------------------------------------------------------------------
# 用例 4：孔的上下偏差（φ30）—— 验证 Δ 换算规则
# --------------------------------------------------------------------------
@pytest.mark.parametrize("letter,grade,ES,EI", [
    ("H", 7, 21.0, 0.0),       # φ30H7：EI = −es(h) = 0，ES = 0 + 21
    ("G", 7, 28.0, 7.0),       # φ30G7：EI = −es(g) = +7
    ("F", 8, 53.0, 20.0),      # φ30F8：EI = −es(f) = +20，ES = 20 + 33
    ("K", 7, 6.0, -15.0),      # φ30K7：ES = −ei(k) + Δ = −2 + 8 = +6
    ("M", 7, 0.0, -21.0),      # φ30M7：ES = −ei(m) + Δ = −8 + 8 = 0
    ("N", 7, -7.0, -28.0),     # φ30N7：ES = −ei(n) + Δ = −15 + 8 = −7
])
def test_hole_deviations(letter, grade, ES, EI):
    """孔的偏差由同名轴偏差换算，J~ZC 要加 Δ = IT_n − IT_(n−1)。

    以 K7 为例手算一遍（这是最容易错的一格）：
      φ30 轴 k 的基本偏差 ei = +2 μm
      Δ = IT7 − IT6 = 21 − 13 = 8 μm
      ES = −ei + Δ = −2 + 8 = +6 μm
      EI = ES − IT7 = 6 − 21 = −15 μm
    若忘了加 Δ，会得到 ES = −2、EI = −23，与标准表对不上。
    """
    res = fits.hole_deviation(30.0, letter, grade)
    assert res["ES_um"] == pytest.approx(ES)
    assert res["EI_um"] == pytest.approx(EI)
    assert res["ES_um"] - res["EI_um"] == pytest.approx(fits.standard_tolerance(30.0, grade))


# --------------------------------------------------------------------------
# 用例 5：同名配合互换性 —— H7/k6 与 K7/h6 必须等价
# --------------------------------------------------------------------------
def test_same_name_fit_equivalence():
    """这是验证 Δ 规则是否正确的**决定性用例**。

    教材结论：基孔制 H7/k6 与基轴制 K7/h6 的极限间隙 / 过盈完全相同。
      H7/k6：孔 (0, +21)，轴 (+2, +15)
         最大间隙 = ES − ei = 21 − 2  = 19 μm
         最小间隙 = EI − es = 0 − 15  = −15 μm（即最大过盈 15 μm）
      K7/h6：孔 (+6, −15)，轴 (0, −13)
         最大间隙 = 6 − (−13) = 19 μm   ✓
         最小间隙 = −15 − 0  = −15 μm   ✓
    两者一致，说明孔 K7 的 Δ 换算做对了。
    """
    base_hole = fits.fit(30.0, "H7", "k6")
    base_shaft = fits.fit(30.0, "K7", "h6")

    assert base_hole["max_clearance_um"] == pytest.approx(19.0)
    assert base_hole["min_clearance_um"] == pytest.approx(-15.0)
    assert base_hole["fit_kind"] == "过渡配合"

    assert base_shaft["max_clearance_um"] == pytest.approx(base_hole["max_clearance_um"])
    assert base_shaft["min_clearance_um"] == pytest.approx(base_hole["min_clearance_um"])
    assert base_shaft["fit_kind"] == base_hole["fit_kind"]


# --------------------------------------------------------------------------
# 用例 6：配合性质判断
# --------------------------------------------------------------------------
def test_fit_kinds():
    """φ30 H7/g6 → 间隙配合；φ30 H7/s6 → 过盈配合。
      H7/g6：最大间隙 = 21 − (−20) = 41 μm，最小间隙 = 0 − (−7) = 7 μm → 间隙
      H7/s6：最大间隙 = 21 − 35 = −14 μm（即最小过盈 14），
             最小间隙 = 0 − 48 = −48 μm（即最大过盈 48） → 过盈
    """
    clear = fits.fit(30.0, "H7", "g6")
    assert clear["fit_kind"] == "间隙配合"
    assert clear["max_clearance_um"] == pytest.approx(41.0)
    assert clear["min_clearance_um"] == pytest.approx(7.0)
    assert clear["base_system"].startswith("基孔制")

    interference = fits.fit(30.0, "H7", "s6")
    assert interference["fit_kind"] == "过盈配合"
    assert interference["min_interference_um"] == pytest.approx(14.0)
    assert interference["max_interference_um"] == pytest.approx(48.0)


# --------------------------------------------------------------------------
# 用例 7：极限尺寸（可以直接抄到图纸上的数）
# --------------------------------------------------------------------------
def test_limit_sizes():
    """φ30 H7/f7：
      孔：30.000 ~ 30.021 mm
      轴：29.980 ~ 29.959 mm
    """
    res = fits.fit(30.0, "H7", "f7")
    assert res["hole_limits_mm"] == pytest.approx([30.000, 30.021], abs=1e-9)
    assert res["shaft_limits_mm"] == pytest.approx([29.959, 29.980], abs=1e-9)


# --------------------------------------------------------------------------
# 用例 8：没有标准值就该报错，绝不给近似值
# --------------------------------------------------------------------------
def test_unsupported_codes_raise():
    """a、b、j、t、u 等代号本项目未内置，必须明确报错而不是给个"差不多"的数。"""
    for code in ("u", "a", "j", "z"):
        with pytest.raises(ValueError):
            fits.shaft_deviation(30.0, code, 6)

    # 尺寸 >80 mm 的 s 没有可核实数据，同样报错
    with pytest.raises(ValueError):
        fits.shaft_deviation(100.0, "s", 6)
    # 超出 500 mm
    with pytest.raises(ValueError):
        fits.shaft_deviation(600.0, "h", 7)
    # 不支持的公差等级
    with pytest.raises(ValueError):
        fits.standard_tolerance(30.0, 3)


# --------------------------------------------------------------------------
# 用例 9：对称公差 js
# --------------------------------------------------------------------------
def test_js_symmetric():
    """φ30 js6：IT6 = 13 μm，故为 ±6.5 μm。"""
    res = fits.shaft_deviation(30.0, "js", 6)
    assert res["es_um"] == pytest.approx(6.5)
    assert res["ei_um"] == pytest.approx(-6.5)
