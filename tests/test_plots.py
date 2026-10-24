"""阶段 3：挠度 / 弯矩分布图测试

【盯什么】
图的数据源是公式库已经过手算对照的 curve（阶段 1 测试兜底），所以这里只盯
绘图层自己引入的三类风险：
  1) **单位换算**：库里 x/y 是 m，图上按表单口径显示 mm——量级错 1000 倍
     在图上反而不容易肉眼看出（曲线形状一样！），必须用数值断言。
     断言值从"用户输入"起手算（换算 + 公式一起算），延续诚实红线。
  2) **中文字体 / 负号**：任务书点名的坑——缺字体设置时中文和悬臂梁的
     负弯矩都会画成方框。这里锁定 rcParams 防回归，字形是否真的渲染出来
     由冒烟检查 stderr 的缺字形警告兜底。
  3) **输出有效性**：页面拿到的是能显示的 PNG（魔数 + 页面内嵌 data URI）。

运行：python -m pytest tests/test_plots.py -v
"""

import matplotlib.pyplot as plt
import pytest

from app import beam_from_form
from plots import beam_curve_series, beam_curves_png, png_data_uri
from calcs import beam as beam_lib

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _center_load_curve():
    """简支梁跨中集中力（阶段 1 经典用例的表单版），供多条用例复用。"""
    return beam_from_form({
        "support": "simply", "L": "2000", "P": "1000",
        "material": "custom", "E_gpa": "200",
        "section": "custom", "I": "100000",
    })


# --------------------------------------------------------------------------
# 1. 换算：m → mm 必须精确（形状一样、量级差 1000 倍的图，肉眼验不出来）
# --------------------------------------------------------------------------
def test_series_conversion_matches_hand_calc():
    """手算：L=2000 mm, P=1000 N, E=200 GPa, I=1e5 mm⁴（→ L=2 m, I=1e-7 m⁴）
      x 轴换算：0 ~ 2000 mm
      挠度换算：y(跨中) = PL³/(48EI) = 8.3333 mm；两端 y = 0
      弯矩不换算：M(跨中) = P·L/4 = 500 N·m；支座处 M = 0
    """
    res = _center_load_curve()
    xs, ys, Ms = beam_curve_series(res["curve"])

    assert xs[0] == pytest.approx(0.0, abs=1e-9)
    assert xs[-1] == pytest.approx(2000.0)               # 2 m → 2000 mm

    center = len(xs) // 2                                # 201 点取中点
    assert xs[center] == pytest.approx(1000.0)           # 跨中位置 mm
    assert ys[center] == pytest.approx(8.3333, rel=1e-4) # 8.3333e-3 m → 8.3333 mm
    assert ys[0] == pytest.approx(0.0, abs=1e-9)
    assert ys[-1] == pytest.approx(0.0, abs=1e-9)        # 简支梁两端挠度为零

    assert Ms[center] == pytest.approx(500.0, rel=1e-6)
    assert Ms[0] == pytest.approx(0.0, abs=1e-9)
    assert Ms[-1] == pytest.approx(0.0, abs=1e-9)


def test_series_cantilever_negative_moment_preserved():
    """手算：悬臂 L=1000 mm, P=500 N, E=200 GPa, I=1e5 mm⁴
      固定端弯矩 = −P·L = −500 N·m（全负值！负号渲染就是靠
      axes.unicode_minus=False 那一行保住的）
      自由端挠度 = PL³/(3EI) = 8.3333 mm
    """
    res = beam_from_form({
        "support": "cantilever", "L": "1000", "P": "500",
        "material": "custom", "E_gpa": "200",
        "section": "custom", "I": "100000",
    })
    xs, ys, Ms = beam_curve_series(res["curve"])

    assert max(Ms) <= 0.0                                # 悬臂梁弯矩全负
    assert min(Ms) == pytest.approx(-500.0, rel=1e-6)    # 固定端
    assert ys[-1] == pytest.approx(8.3333, rel=1e-4)     # 自由端挠度最大
    assert xs[-1] == pytest.approx(1000.0)


# --------------------------------------------------------------------------
# 2. 中文字体与负号设置（任务书点名的坑，锁死防回归）
# --------------------------------------------------------------------------
def test_cjk_font_and_minus_sign_configured():
    assert plt.rcParams["axes.unicode_minus"] is False
    font_list = plt.rcParams["font.sans-serif"]
    # 捆绑的 Noto Sans SC 注册后应置于候选链首位（开发与线上字体统一）
    assert font_list[0] == "Noto Sans SC"
    import os
    assert os.path.exists(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "fonts", "NotoSansSC-Regular.otf",
    ))


def test_bundled_font_alone_renders_cjk_and_minus_without_warnings():
    """模拟 Render 环境：服务器上没有任何系统中文字体，候选链只剩捆绑的
    Noto Sans SC。悬臂梁（弯矩全负）+ 全中文标注，任何缺字形警告都算失败。

    这是阶段 4 上线前最重要的一条防线——线上没有第二道兜底。
    """
    import warnings

    res = beam_from_form({
        "support": "cantilever", "L": "1000", "P": "500",
        "material": "custom", "E_gpa": "200",
        "section": "custom", "I": "100000",
    })
    with warnings.catch_warnings():
        warnings.simplefilter("error")              # 缺字形警告 → 当场抛错
        plt.rcParams["font.sans-serif"] = ["Noto Sans SC"]   # 只给捆绑字体
        try:
            png = beam_curves_png(
                res["curve"], res["support"],
                max_deflection_mm=res["max_deflection_mm"],
                max_deflection_x_mm=res["max_deflection_at_x_mm"],
                max_moment_Nm=res["max_moment_Nm"],
                max_moment_x_mm=res["max_moment_at_x_mm"],
            )
        finally:
            plt.rcParams["font.sans-serif"] = ["Noto Sans SC"] + [
                "Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                "WenQuanYi Zen Hei", "sans-serif",
            ]
    assert png.startswith(PNG_MAGIC)


# --------------------------------------------------------------------------
# 3. PNG 输出有效性
# --------------------------------------------------------------------------
def test_png_output_is_valid_and_annotated():
    res = _center_load_curve()
    png = beam_curves_png(
        res["curve"], res["support"],
        max_deflection_mm=res["max_deflection_mm"],
        max_deflection_x_mm=res["max_deflection_at_x_mm"],
        max_moment_Nm=res["max_moment_Nm"],
        max_moment_x_mm=res["max_moment_at_x_mm"],
    )
    assert png.startswith(PNG_MAGIC)
    assert len(png) > 5000                               # 空图/白图不会有这个体量


def test_data_uri_embeddable():
    png = beam_curves_png(
        beam_lib.simply_supported(L=2.0, E=2e11, I=1e-7, P=1000.0)["curve"], "简支梁",
    )
    uri = png_data_uri(png)
    assert uri.startswith("data:image/png;base64,")
    # base64 解码后必须还原成同一个 PNG（防编码环节出幺蛾子）
    import base64
    assert base64.b64decode(uri.split(",", 1)[1]) == png


def test_beam_page_embeds_plot_after_post(client):
    """端到端：POST 梁表单 → 结果页必须内嵌 <img> 的 data URI 图。"""
    resp = client.post("/beam", data={
        "support": "cantilever", "L": "1000", "P": "500",
        "material": "custom", "E_gpa": "200",
        "section": "custom", "I": "100000",
    })
    body = resp.get_data(as_text=True)
    assert resp.status_code == 200
    assert 'class="plot"' in body
    assert "data:image/png;base64," in body
