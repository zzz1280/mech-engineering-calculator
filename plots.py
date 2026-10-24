"""梁的挠度 / 弯矩分布图（阶段 3）—— 服务端 Matplotlib 出图

【为什么这个文件在 Web 层、而不进 calcs/】
图是"展示物"，公式库必须保持纯函数零改动（阶段 1 的 49 项测试是它的安全网）。
曲线数据由库里 simply_supported / cantilever 的返回值 `curve` 给出
（SI 单位：x/y 为 m，M 为 N·m），本模块只做两件事：
  1. **单位换算**：x、y 换成 mm（与本站表单口径一致），M 保持 N·m；
  2. 画图并输出 PNG 字节流。

【中文与负号显示（任务书点名的坑）】
  · 中文字体：Render 等云端 Linux 环境**不预装任何 CJK 字体**，只写候选列表
    到时候照样全是方框。所以仓库捆绑了 Noto Sans SC（OFL 开源许可，可再分发；
    Windows 自带的微软雅黑有版权，不能进仓库），启动时 addfont 注册并置于
    候选链首位——开发和线上渲染出的图字体一致。
  · 负号：axes.unicode_minus = False，强制用 ASCII 连字符。
    不关它的话，Matplotlib 会输出 U+2212（数学负号），中文字体里没有这个字形，
    图上所有负数（悬臂梁弯矩就是负的！）会渲染成方框 □。

【无界面后端】
    import 后立刻 matplotlib.use("Agg")——服务器上没有显示器，
    默认后端会尝试弹窗，测试环境更是直接崩。必须在 import pyplot 之前切换。
"""

from __future__ import annotations

import base64
import os
from io import BytesIO
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")                       # 必须在 pyplot 之前：无界面后端
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 捆绑的开源中文字体（fonts/NotoSansSC-Regular.otf，OFL 许可）。
# 存在就注册到 matplotlib 的字体管理器并置于候选链首位，
# 这样 Render 这类没有预装中文字体的环境也能正常出图。
_BUNDLED_FONT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "fonts", "NotoSansSC-Regular.otf")
_FONT_CANDIDATES = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "sans-serif",
]
if os.path.exists(_BUNDLED_FONT):
    font_manager.fontManager.addfont(_BUNDLED_FONT)
    plt.rcParams["font.sans-serif"] = ["Noto Sans SC"] + _FONT_CANDIDATES
else:                                       # 字体文件缺失时退回系统候选链，不崩
    plt.rcParams["font.sans-serif"] = _FONT_CANDIDATES
plt.rcParams["axes.unicode_minus"] = False

_COLOR_DEFLECTION = "#0f5c8c"
_COLOR_MOMENT = "#b45309"


def beam_curve_series(curve: List[Dict[str, float]]) -> Tuple[List[float], List[float], List[float]]:
    """把库返回的 curve（SI）整理成画图用的三条序列。

    curve 每个元素形如 {"x": m, "M": N·m, "y": m}
    返回 (xs_mm, ys_mm, Ms_Nm)：
        xs_mm  位置，m → mm（×1000）
        ys_mm  挠度，m → mm（×1000）；正 = 向下（与载荷同向，见 calcs/beam.py 约定）
        Ms_Nm  弯矩，N·m 原样；正 = 下侧受拉
    换算只在这一处发生，测试对它做手算对照。
    """
    xs_mm = [p["x"] * 1000.0 for p in curve]
    ys_mm = [p["y"] * 1000.0 for p in curve]
    Ms_Nm = [p["M"] for p in curve]
    return xs_mm, ys_mm, Ms_Nm


def beam_curves_png(
    curve: List[Dict[str, float]],
    support: str,
    *,
    max_deflection_mm: float | None = None,
    max_deflection_x_mm: float | None = None,
    max_moment_Nm: float | None = None,
    max_moment_x_mm: float | None = None,
) -> bytes:
    """画"挠度 + 弯矩"两联分布图，返回 PNG 字节流。

    curve   : 库返回的采样曲线（SI），201 个点足够平滑
    support : "简支梁" / "悬臂梁"，只用于标题
    其余参数是结果页表格里已有的极值点，标注到图上与曲线互相印证。
    """
    xs, ys, Ms = beam_curve_series(curve)

    fig, (ax_y, ax_m) = plt.subplots(
        2, 1, figsize=(7.2, 5.8), sharex=True, dpi=110,
        gridspec_kw={"hspace": 0.12},
    )

    # ---- 上图：挠度 y(x)，正方向向下（与载荷同向）----
    ax_y.plot(xs, ys, color=_COLOR_DEFLECTION, lw=1.8)
    ax_y.fill_between(xs, ys, 0.0, color=_COLOR_DEFLECTION, alpha=0.08)
    ax_y.axhline(0.0, color="#888888", lw=0.8)
    ax_y.set_ylabel("挠度 y (mm)\n正 = 向下")
    ax_y.grid(True, alpha=0.3)
    if max_deflection_mm is not None and max_deflection_x_mm is not None:
        ax_y.annotate(
            f"δmax = {max_deflection_mm:.4g} mm\nx = {max_deflection_x_mm:.4g} mm",
            xy=(max_deflection_x_mm, max_deflection_mm),
            xytext=(12, -4), textcoords="offset points",
            fontsize=9, color=_COLOR_DEFLECTION,
        )

    # ---- 下图：弯矩 M(x)，正 = 下侧受拉；悬臂梁全负值依赖 ASCII 负号 ----
    ax_m.plot(xs, Ms, color=_COLOR_MOMENT, lw=1.8)
    ax_m.axhline(0.0, color="#888888", lw=0.8)   # 零弯矩线：极值在剪力为零处
    ax_m.set_ylabel("弯矩 M (N·m)\n正 = 下侧受拉")
    ax_m.set_xlabel("位置 x (mm)（简支梁从左支座量起，悬臂梁从固定端量起）")
    ax_m.grid(True, alpha=0.3)
    if max_moment_Nm is not None and max_moment_x_mm is not None:
        ax_m.annotate(
            f"Mmax = {max_moment_Nm:.4g} N·m",
            xy=(max_moment_x_mm, max_moment_Nm),
            xytext=(12, 0), textcoords="offset points",
            fontsize=9, color=_COLOR_MOMENT,
        )

    ax_y.set_title(f"{support}：挠度与弯矩分布图", fontsize=11)

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)                              # 不关的话每次请求泄漏一个 Figure
    return buf.getvalue()


def png_data_uri(png_bytes: bytes) -> str:
    """PNG 字节流 → data URI，直接塞进 <img src>。

    为什么不用存文件：站点无状态、多 worker 部署（阶段 4）时
    临时文件的读写与清理都是麻烦，内嵌 data URI 天然没这些问题。
    """
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
