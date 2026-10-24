"""螺栓组受力 —— 手算对照测试

【这组用例要盯的概念】
1) 倾覆力矩公式 F_i = M·L_i/ΣL_j² 的前提是"接合面保持平面"，
   于是螺栓受力与到翻转轴线的距离成正比 —— 离轴线越远越危险。
2) 倾覆力矩用的是**到翻转轴线**的距离，不是到螺栓组形心的距离！
   这是教材里最容易和"受扭矩"搞混的地方，本文件的用例特意把两种工况分开。
"""

import pytest

from calcs import bolt_group


# --------------------------------------------------------------------------
# 用例 1：轴向载荷通过形心 —— 各螺栓均分
# --------------------------------------------------------------------------
def test_axial_uniform():
    """手算：总轴向载荷 F = 12000 N，4 个螺栓均布
      F_i = F/z = 12000/4 = 3000 N
    """
    res = bolt_group.axial_uniform(F=12000.0, z=4)

    assert res["loads_N"] == pytest.approx([3000.0] * 4)
    assert res["max_load_N"] == pytest.approx(3000.0)


# --------------------------------------------------------------------------
# 用例 2：纯倾覆力矩 —— 两排螺栓，远排更危险
# --------------------------------------------------------------------------
def test_overturning_two_rows():
    """手算：4 个螺栓，分两排（每排 2 个）
      远排距翻转轴线 L = 200 mm，近排 L = 50 mm，M = 1.0×10⁶ N·mm

      ΣL² = 2×200² + 2×50² = 2×40000 + 2×2500 = 80000 + 5000 = 85000 mm²
      远排单个螺栓：F = M·L/ΣL² = 1e6 × 200 / 85000 = 2352.94 N
      近排单个螺栓：F = 1e6 × 50 / 85000  = 588.24 N
      校核：2×2352.94 + 2×588.24 = 4705.88 + 1176.47 = 5882.35 N·m... 
      力矩平衡校核：Σ(F_i·L_i) = 2×2352.94×200 + 2×588.24×50
                  = 941176 + 58824 = 1.0e6 N·mm = M ✓（说明分配正确）
    """
    res = bolt_group.overturning_about_axis(
        M=1.0e6, distances=[200.0, 200.0, 50.0, 50.0]
    )

    assert res["sum_distance_squared_mm2"] == pytest.approx(85000.0)
    assert res["max_load_N"] == pytest.approx(2352.941, rel=1e-5)
    # 最危险的是距离最大的螺栓（前两个中的任一个）
    assert res["max_load_bolt"] in ("1", "2")

    loads = [b["total_N"] for b in res["bolts"]]
    assert loads[0] == pytest.approx(2352.941, rel=1e-5)
    assert loads[2] == pytest.approx(588.235, rel=1e-5)

    # 反算：各螺栓对翻转轴线的力矩之和，应等于外加倾覆力矩（能量/静力学校核）
    moment_sum = sum(b["total_N"] * b["distance_mm"] for b in res["bolts"])
    assert moment_sum == pytest.approx(1.0e6, rel=1e-9)


# --------------------------------------------------------------------------
# 用例 3：倾覆力矩 + 轴向载荷叠加
# --------------------------------------------------------------------------
def test_overturning_plus_axial():
    """承接用例 2，再叠加总轴向载荷 F = 8000 N（沿螺栓轴线，通过形心）
      每个螺栓先均分：8000/4 = 2000 N
      远排（受拉侧）：2000 + 2352.94 = 4352.94 N
      近排：2000 + 588.24 = 2588.24 N
    """
    res = bolt_group.overturning_about_axis(
        M=1.0e6, distances=[200.0, 200.0, 50.0, 50.0], F_axial=8000.0
    )

    assert res["max_load_N"] == pytest.approx(4352.941, rel=1e-5)
    assert res["bolts"][2]["total_N"] == pytest.approx(2588.235, rel=1e-5)
    assert res["bolts"][0]["from_axial_N"] == pytest.approx(2000.0)


# --------------------------------------------------------------------------
# 用例 4：圆形凸缘（法兰）均布螺栓 —— 闭式解 F_max = 4M/(z·D)
# --------------------------------------------------------------------------
def test_circular_flange_z6():
    """手算：6 个螺栓均布于 D = 200 mm 的中心圆（r = 100 mm），M = 5×10⁵ N·mm

      螺栓角度 θ = 0°, 60°, 120°, 180°, 240°, 300°
      到翻转轴线的距离 L_i = r·|sin θ_i|
                          = 0, 86.60, 86.60, 0, 86.60, 86.60 mm
      ΣL² = 4 × 86.60² = 30000 mm²（也等于 z·r²/2 = 6×100²/2 = 30000 ✓）
      F_max = M·L_max/ΣL² = 5e5 × 86.60 / 30000 = 1443.38 N

      ⚠️ 这里不能套 F_max = 4M/(z·D)：那会算出 1666.67 N，**高估 15.5%**。
      原因见 calcs/bolt_group.py 的说明 —— z=6 时没有任何螺栓落在离轴线最远处。
    """
    res = bolt_group.overturning_circular_flange(M=5.0e5, z=6, D=200.0)

    assert res["sum_distance_squared_mm2"] == pytest.approx(30000.0, rel=1e-9)
    assert res["max_distance_mm"] == pytest.approx(86.6025, rel=1e-5)
    assert res["max_load_N"] == pytest.approx(1443.376, rel=1e-5)
    assert res["closed_form_max_N"] == pytest.approx(1443.376, rel=1e-5)
    # 明确断言"不是"那个想当然的值，防止以后有人把它改回去
    assert res["max_load_N"] != pytest.approx(4 * 5.0e5 / (6 * 200.0), rel=1e-3)


def test_circular_flange_z4_closed_form_holds():
    """对照用例：z = 4 时，θ = 0°,90°,180°,270°，恰好有螺栓落在最远处（L_max = r），
    这时简化的闭式 F_max = 4M/(z·D) 才成立。

      手算：M = 5×10⁵ N·mm, z = 4, D = 200 mm
        L_i = 0, 100, 0, 100 mm；ΣL² = 2×100² = 20000 = z·r²/2 ✓
        F_max = 5e5 × 100 / 20000 = 2500 N
        4M/(z·D) = 4×5e5/(4×200) = 2500 N ✓ 两式吻合
    """
    res = bolt_group.overturning_circular_flange(M=5.0e5, z=4, D=200.0)

    assert res["max_distance_mm"] == pytest.approx(100.0, rel=1e-9)
    assert res["max_load_N"] == pytest.approx(2500.0, rel=1e-9)
    assert res["closed_form_max_N"] == pytest.approx(2500.0, rel=1e-9)


# --------------------------------------------------------------------------
# 用例 5：铰制孔螺栓受扭矩（与倾覆力矩区分开！）
# --------------------------------------------------------------------------
def test_torque_circular():
    """手算：6 个铰制孔螺栓均布于 D = 120 mm 圆周，传递扭矩 T = 1.2×10⁵ N·mm
      F = 2T/(z·D) = 2×1.2e5 / (6×120) = 2.4e5/720 = 333.33 N
      校核：Σ(F·r) = 6 × 333.33 × 60 = 1.2e5 N·mm = T ✓
    """
    res = bolt_group.torque_circular(T=1.2e5, z=6, D=120.0)

    assert res["max_load_N"] == pytest.approx(333.333, rel=1e-5)
    torque_sum = 6 * 333.333 * 60.0
    assert torque_sum == pytest.approx(1.2e5, rel=1e-4)


# --------------------------------------------------------------------------
# 用例 6：普通螺栓靠摩擦承受横向载荷 —— 求所需预紧力
# --------------------------------------------------------------------------
def test_transverse_friction_preload():
    """手算：横向载荷 F_Σ = 5000 N，4 个螺栓，f = 0.15，单剪(m=1)，K_f = 1.2
      F_0 ≥ K_f·F_Σ/(f·z·m) = 1.2×5000 / (0.15×4×1) = 6000/0.6 = 10000 N
      注意这个 10000 N 远大于 5000/4 = 1250 N —— 这正是"靠摩擦"的代价。
    """
    res = bolt_group.transverse_friction_preload(
        F_transverse=5000.0, z=4, friction_coef=0.15, joints=1, safety_factor=1.2
    )

    assert res["required_preload_N"] == pytest.approx(10000.0)


# --------------------------------------------------------------------------
# 用例 7：错误输入
# --------------------------------------------------------------------------
def test_invalid_input_raises():
    with pytest.raises(ValueError):
        bolt_group.axial_uniform(F=1000.0, z=0)                # 没有螺栓
    with pytest.raises(ValueError):
        bolt_group.overturning_about_axis(M=1e6, distances=[0.0, 0.0])  # 全都压在轴线上
    with pytest.raises(ValueError):
        bolt_group.overturning_circular_flange(M=1e6, z=2, D=100.0)     # z<3 闭式不成立
