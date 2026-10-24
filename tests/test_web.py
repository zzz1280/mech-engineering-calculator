"""Flask Web 层测试（阶段 2）

【这组测试盯什么】
阶段 1 的测试盯"公式抄没抄错"；这组测试盯"Web 层有没有把好事办坏"：
  1) **单位换算链**：用户填的是 mm / N/mm / GPa / mm⁴，换算成 SI 后再进公式库——
     换算错一个量级，公式库再对也算错。所以每条用例仍然用手算值断言
     （手算从"用户输入值"开始，把换算和公式一起算完），不是拿库输出对库输出。
  2) **脏输入的友好拦截**：空值、非数字、越界必须得到中文提示，
     绝不允许 500 / Traceback 出现在用户面前。
  3) **公式库自身的 ValueError 原样透传**：比如公差代号未内置——
     这类"诚实报错"消息写得很好，Web 层不该吞掉或改写。
  4) 页面可达、结果与依据同页呈现。

运行：python -m pytest tests/test_web.py -v
"""

import math

import pytest

from app import (
    beam_from_form,
    bearing_from_form,
    bolt_from_form,
    fits_deviation_from_form,
    fits_fit_from_form,
    sig,
)
from web_inputs import FormError


# --------------------------------------------------------------------------
# 1. 页面可达性
# --------------------------------------------------------------------------
@pytest.mark.parametrize("path,keyword", [
    ("/", "机械工程计算器"),
    ("/beam", "梁的挠度与弯矩"),
    ("/bolt-group", "螺栓组受力分析"),
    ("/bearing", "深沟球轴承寿命"),
    ("/fits", "公差与配合查询"),
])
def test_pages_get_200(client, path, keyword):
    resp = client.get(path)
    assert resp.status_code == 200
    assert keyword in resp.get_data(as_text=True)


# --------------------------------------------------------------------------
# 2. 梁计算器：换算链 + 手算对照
# --------------------------------------------------------------------------
def test_beam_web_center_point_load_matches_hand_calc():
    """手算（从用户输入开始，含换算）：
      输入 L=2000 mm, P=1000 N, 自定义 E=200 GPa, I=100000 mm⁴
      换算：L=2 m, E=2e11 Pa, I=1e5×1e-12=1e-7 m⁴
      δ_max = PL³/(48EI) = 1000×8/(48×2e11×1e-7) = 8000/960000 = 8.3333e-3 m = 8.333 mm
      M_max = P·L/4 = 500 N·m，R_左 = R_右 = 500 N
    """
    res = beam_from_form({
        "support": "simply", "L": "2000", "P": "1000",
        "material": "custom", "E_gpa": "200",
        "section": "custom", "I": "100000",
    })
    assert res["max_deflection_mm"] == pytest.approx(8.3333, rel=1e-4)
    assert res["max_moment_Nm"] == pytest.approx(500.0, rel=1e-6)
    assert res["reactions"]["left_N"] == pytest.approx(500.0)
    # 换算回显里应能看到 2000 mm → 2 m
    assert any("2 m" in si for _, _, si in res["si_echo"])


def test_beam_web_uniform_load_q_conversion():
    """手算：L=2000 mm, q=1 N/mm（→ 1000 N/m）, E=200 GPa, I=1e5 mm⁴
      δ_max = 5qL⁴/(384EI) = 5×1000×16/(384×2e11×1e-7) = 80000/7.68e6 = 10.4167 mm
      M_max = q·L²/8 = 1000×4/8 = 500 N·m
      q 的换算（N/mm → N/m ×1000）是最容易错的一步，单独验一次。
    """
    res = beam_from_form({
        "support": "simply", "L": "2000", "q": "1",
        "material": "custom", "E_gpa": "200",
        "section": "custom", "I": "100000",
    })
    assert res["max_deflection_mm"] == pytest.approx(10.4167, rel=1e-4)
    assert res["max_moment_Nm"] == pytest.approx(500.0, rel=1e-6)


def test_beam_web_rect_section():
    """手算：L=1000 mm（=1 m）, P=100 N, E=200 GPa, 矩形 b=20 h=50 mm
      I = b·h³/12 = 20×125000/12 = 208333.33 mm⁴ = 2.083333e-7 m⁴
      δ_max = PL³/(48EI) = 100/(48×2e11×2.08333e-7) = 100/2e6 = 5e-5 m = 0.05 mm
      M_max = P·L/4 = 25 N·m
    """
    res = beam_from_form({
        "support": "simply", "L": "1000", "P": "100",
        "material": "custom", "E_gpa": "200",
        "section": "rect", "b": "20", "h": "50",
    })
    assert res["max_deflection_mm"] == pytest.approx(0.05, rel=1e-4)
    assert res["max_moment_Nm"] == pytest.approx(25.0, rel=1e-6)


def test_beam_web_circle_section_with_material_steel():
    """手算：L=2000 mm, P=1000 N, 材料=钢（库内 ELASTIC_MODULUS=2.06e11 Pa）,
      实心圆 d=40 mm：I = π×(0.04)⁴/64 = 1.256637e-7 m⁴
      δ_max = 1000×8/(48×2.06e11×1.256637e-7) = 8000/1.242553e6 = 6.4384 mm
    """
    res = beam_from_form({
        "support": "simply", "L": "2000", "P": "1000",
        "material": "钢",
        "section": "circle", "d": "40",
    })
    expected_mm = 1000.0 * 8.0 / (
        48.0 * 2.06e11 * (math.pi * 0.04 ** 4 / 64.0)
    ) * 1000.0
    assert res["max_deflection_mm"] == pytest.approx(expected_mm, rel=1e-6)


def test_beam_web_off_center_load_position_conversion():
    """手算（阶段 1 用例 5 换成表单输入）：
      L=3000 mm, P=900 N, a=1000 mm → 换算后与阶段 1 相同工况
      δ_max = 21.773 mm，在 x=1367.01 mm；M_max = 600 N·m
    """
    res = beam_from_form({
        "support": "simply", "L": "3000", "P": "900", "a": "1000",
        "material": "custom", "E_gpa": "200",
        "section": "custom", "I": "100000",
    })
    assert res["max_deflection_mm"] == pytest.approx(21.773, rel=1e-3)
    assert res["max_deflection_at_x_mm"] == pytest.approx(1367.01, abs=30.0)
    assert res["max_moment_Nm"] == pytest.approx(600.0, rel=1e-6)


def test_beam_web_rejects_all_zero_loads():
    with pytest.raises(FormError, match="至少输入一种载荷"):
        beam_from_form({
            "support": "simply", "L": "2000", "P": "0", "q": "0",
            "material": "钢", "section": "custom", "I": "100000",
        })


def test_beam_web_rejects_non_numeric():
    with pytest.raises(FormError, match="请输入数字"):
        beam_from_form({
            "support": "simply", "L": "两米", "P": "1000",
            "material": "钢", "section": "custom", "I": "100000",
        })


def test_beam_web_load_position_out_of_range_is_friendly_mm():
    """a 超出跨度：报错必须用用户输入的 mm 说话（库内报错是 m，不能透传给用户）。"""
    with pytest.raises(FormError, match="2000"):
        beam_from_form({
            "support": "simply", "L": "2000", "P": "1000", "a": "2500",
            "material": "钢", "section": "custom", "I": "100000",
        })


# --------------------------------------------------------------------------
# 3. 螺栓组计算器：五种工况全走一遍（数值沿用阶段 1 手算用例）
# --------------------------------------------------------------------------
def test_bolt_web_axial_uniform():
    """手算：F=12000 N, z=4 → 每个 3000 N"""
    res = bolt_from_form({"case": "axial", "F": "12000", "z": "4"})
    assert res["loads_N"] == pytest.approx([3000.0] * 4)


def test_bolt_web_overturning_distance_list_parsing():
    """手算：M=1e6 N·mm，距离 "200, 200, 50, 50"（字符串！Web 层要会拆）
      ΣL² = 85000 mm²，远排 2352.94 N，近排 588.24 N
    """
    res = bolt_from_form({
        "case": "overturn", "M": "1000000",
        "distances": "200, 200, 50, 50", "F_axial": "",
    })
    assert res["sum_distance_squared_mm2"] == pytest.approx(85000.0)
    assert res["max_load_N"] == pytest.approx(2352.941, rel=1e-5)
    assert res["bolts"][2]["total_N"] == pytest.approx(588.235, rel=1e-5)


def test_bolt_web_overturning_plus_axial():
    """手算：叠加 F_axial=8000 N → 远排 4352.94 N，近排 2588.24 N"""
    res = bolt_from_form({
        "case": "overturn", "M": "1000000",
        "distances": "200, 200, 50, 50", "F_axial": "8000",
    })
    assert res["max_load_N"] == pytest.approx(4352.941, rel=1e-5)


def test_bolt_web_circular_flange_z6():
    """手算：M=5e5, z=6, D=200 → F_max = 1443.376 N（不是 4M/(zD)=1666.67！）"""
    res = bolt_from_form({"case": "flange", "M": "500000", "z": "6", "D": "200", "F_axial": ""})
    assert res["max_load_N"] == pytest.approx(1443.376, rel=1e-5)
    assert res["max_distance_mm"] == pytest.approx(86.6025, rel=1e-5)


def test_bolt_web_torque():
    """手算：T=1.2e5 N·mm, z=6, D=120 → F = 2T/(zD) = 333.333 N"""
    res = bolt_from_form({"case": "torque", "T": "120000", "z": "6", "D": "120"})
    assert res["max_load_N"] == pytest.approx(333.333, rel=1e-5)


def test_bolt_web_transverse_preload():
    """手算：F_Σ=5000, z=4, f=0.15, m=1, K_f=1.2 → F_0 = 10000 N"""
    res = bolt_from_form({
        "case": "preload", "F_transverse": "5000", "z": "4",
        "f": "0.15", "joints": "1", "K_f": "1.2",
    })
    assert res["required_preload_N"] == pytest.approx(10000.0)


def test_bolt_web_rejects_all_zero_overturning():
    with pytest.raises(FormError, match="倾覆力矩"):
        bolt_from_form({"case": "overturn", "M": "0", "distances": "200, 200", "F_axial": "0"})


# --------------------------------------------------------------------------
# 4. 轴承寿命计算器
# --------------------------------------------------------------------------
def test_bearing_web_integrated_matches_hand_calc():
    """手算（阶段 1 用例 6 的表单版）：
      Fr=4000, Fa=0, Cr=20000, C0r=15000, n=1000 → P=4000 N，
      L10 = (20000/4000)³ = 125×10⁶ 转，L10h = 2083.33 h
    """
    res = bearing_from_form({
        "Fr": "4000", "Fa": "0", "Cr": "20000", "Cor": "15000",
        "n": "1000", "f_t": "", "f_p": "",
    })
    assert res["equivalent_load_N"] == pytest.approx(4000.0)
    assert res["L10_million_rev"] == pytest.approx(125.0)
    assert res["L10_hours"] == pytest.approx(2083.333, rel=1e-5)


def test_bearing_web_with_load_factor():
    """手算：f_p=1.5 → L10h = 2083.33/1.5³ = 617.28 h"""
    res = bearing_from_form({
        "Fr": "4000", "Fa": "0", "Cr": "20000", "Cor": "15000",
        "n": "1000", "f_t": "", "f_p": "1.5",
    })
    assert res["L10_hours"] == pytest.approx(617.284, rel=1e-5)


def test_bearing_web_rejects_zero_loads():
    with pytest.raises(FormError, match="不能同时为 0"):
        bearing_from_form({
            "Fr": "0", "Fa": "0", "Cr": "20000", "Cor": "15000", "n": "1000",
        })


# --------------------------------------------------------------------------
# 5. 公差配合计算器
# --------------------------------------------------------------------------
def test_fits_web_hole_deviation_h7():
    """手算：φ30 H7 → ES=+21 μm, EI=0（GB/T 1800.2 表值）"""
    res = fits_deviation_from_form({"kind": "hole", "nominal": "30", "letter": "H", "grade": "7"})
    assert res["ES_um"] == pytest.approx(21.0)
    assert res["EI_um"] == pytest.approx(0.0)


def test_fits_web_shaft_deviation_g6():
    """手算：φ30 g6 → es=−7 μm, ei=−20 μm"""
    res = fits_deviation_from_form({"kind": "shaft", "nominal": "30", "letter": "g", "grade": "6"})
    assert res["es_um"] == pytest.approx(-7.0)
    assert res["ei_um"] == pytest.approx(-20.0)


def test_fits_web_fit_h7_g6():
    """手算：φ30 H7/g6 → X_max = 21−(−20) = 41 μm，X_min = 0−(−7) = 7 μm，间隙配合"""
    res = fits_fit_from_form({"nominal": "30", "hole_code": "H7", "shaft_code": "g6"})
    assert res["fit_kind"] == "间隙配合"
    assert res["max_clearance_um"] == pytest.approx(41.0)
    assert res["min_clearance_um"] == pytest.approx(7.0)


def test_fits_web_unsupported_letter_passes_through_library_error():
    """代号 'a' 未内置：库的"诚实报错"必须原样透传（消息含"暂未内置"），不能 500。"""
    with pytest.raises(ValueError, match="暂未内置"):
        fits_deviation_from_form({"kind": "shaft", "nominal": "30", "letter": "a", "grade": "7"})


def test_fits_web_bad_code_format():
    with pytest.raises(FormError, match="格式"):
        fits_fit_from_form({"nominal": "30", "hole_code": "H?", "shaft_code": "g6"})


def test_fits_web_grade_out_of_built_range():
    """IT4 未内置：Web 层直接拦截（本工具覆盖 IT5~IT18）。"""
    with pytest.raises(FormError, match="不能小于 5"):
        fits_deviation_from_form({"kind": "hole", "nominal": "30", "letter": "H", "grade": "4"})


# --------------------------------------------------------------------------
# 6. HTTP 层：错误必须变成页面上的友好提示，绝不是 500
# --------------------------------------------------------------------------
def test_client_post_invalid_input_shows_friendly_error(client):
    resp = client.post("/beam", data={"support": "simply", "L": "", "P": "", "q": ""})
    assert resp.status_code == 200                       # 不是 500
    body = resp.get_data(as_text=True)
    assert "请填写" in body                              # 有中文提示
    assert "Traceback" not in body


def test_client_post_bearing_error_keeps_form_page(client):
    resp = client.post("/bearing", data={"Fr": "0", "Fa": "0", "Cr": "", "Cor": "", "n": ""})
    assert resp.status_code == 200
    assert "不能同时为 0" in resp.get_data(as_text=True)


def test_client_post_fits_unsupported_code_shown_on_page(client):
    resp = client.post("/fits", data={
        "mode": "deviation", "kind": "shaft", "nominal": "30", "letter": "a", "grade": "7",
    })
    assert resp.status_code == 200
    assert "暂未内置" in resp.get_data(as_text=True)


def test_client_beam_post_valid_result_on_same_page(client):
    """任务书要求"参数输入表单 → 即时计算结果 + 计算依据"：结果与依据必须同页。"""
    resp = client.post("/beam", data={
        "support": "simply", "L": "2000", "P": "1000",
        "material": "custom", "E_gpa": "200",
        "section": "custom", "I": "100000",
    })
    body = resp.get_data(as_text=True)
    assert resp.status_code == 200
    assert "8.333" in body                               # δ_max = 8.333 mm
    assert "500" in body                                 # M_max = 500 N·m
    assert "计算依据" in body


def test_client_bolt_group_case_dispatch(client):
    """同一页面五个工况小表单：提交哪个就显示哪个的结果。"""
    resp = client.post("/bolt-group", data={"case": "torque", "T": "120000", "z": "6", "D": "120"})
    body = resp.get_data(as_text=True)
    assert resp.status_code == 200
    assert "333.3" in body
    # 其它工况的标题仍在（单页多工况），但不该串出别的结果
    assert "轴向载荷通过形心" in body


# --------------------------------------------------------------------------
# 7. 显示层小工具：有效数字
# --------------------------------------------------------------------------
def test_sig_filter():
    assert sig(2352.941) == "2353"          # 4 位有效数字
    assert sig(8.33333) == "8.333"
    assert sig(500.0) == "500"              # 整数值不带小数点
    assert sig(None) == "—"                 # 无闭式解等场景
    assert sig(21) == "21"
    assert sig(0.000012345) == "1.234e-05"  # 小数转科学计数，不丢有效数字
