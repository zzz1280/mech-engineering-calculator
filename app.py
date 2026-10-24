"""机械工程计算器 —— Flask Web 层（阶段 2）

【分层结构——为什么这样切】
    浏览器表单（字符串）
        → web_inputs.py：脏输入拦截（中文报错，带字段名）
        → app.py 各 *_from_form()：**单位换算到 SI** + 调用 calcs/ 公式库
        → calcs/：纯函数计算（本项目一行都不改，阶段 1 的 49 项测试就是它的安全网）
        → 结果 dict → Jinja2 模板渲染（结果 + 计算依据）

【本页输入单位约定（输入输出层的"换算只发生在这里"）】
    梁计算器     长度 mm、集中力 N、均布载荷 N/mm、模量 GPa（或材料下拉）、惯性矩 mm⁴
    螺栓组       力 N、力矩/扭矩 N·mm、长度 mm   ← 教材口径，与 calcs/bolt_group.py 一致，无需换算
    轴承寿命     载荷 N、转速 r/min              ← 寿命公式直接要 r/min，与库一致，无需换算
    公差配合     尺寸 mm、偏差 μm                ← 公差行业惯例，与库一致，无需换算

    只有梁计算器需要真正的换算（mm→m、N/mm→N/m、GPa→Pa、mm⁴→m⁴），
    换算前后都回显在结果里（si_echo），让"单位混乱"这个头号 bug 来源暴露在阳光下。

【技术栈纪律（任务书范围控制）】
    只用 Flask + Jinja2 服务端渲染，不用前端框架、不用数据库、不写 AJAX——
    表单 POST 回同一 URL、原页显示结果，就是任务书要的"参数输入 → 即时结果 + 计算依据"。
"""

from __future__ import annotations

import os
import re

from flask import Flask, render_template, request

from calcs import ELASTIC_MODULUS
from calcs import beam as beam_lib
from calcs import bolt_group as bolt_lib
from calcs import bearing as bearing_lib
from calcs import fits as fits_lib
from plots import beam_curves_png, png_data_uri
from web_inputs import FormError, parse_choice, parse_float, parse_float_list, parse_int


# --------------------------------------------------------------------------
# 显示层：有效数字规范（任务书加分项）——统一 4 位有效数字
# --------------------------------------------------------------------------
def sig(value, digits: int = 4) -> str:
    """按有效数字格式化数字：2352.941 → '2353'，0.000012345 → '1.234e-05'。

    None（如"该工况无闭式解"）显示为占位符；整数原样，不带小数点。
    """
    if value is None:
        return "—"
    if isinstance(value, int):
        return str(value)
    return f"%.{digits}g" % value


# --------------------------------------------------------------------------
# 梁计算器：表单（mm/N/N/mm/GPa/mm⁴）→ SI → 公式库
# --------------------------------------------------------------------------
def beam_from_form(form) -> dict:
    """解析梁计算表单并调用公式库。返回结果 dict（附带换算回显 si_echo）。"""
    support = parse_choice(form, "support", "支承形式", ("simply", "cantilever"))
    L_mm = parse_float(form, "L", "跨度 L (mm)", min=0, exclusive_min=True)

    P = parse_float(form, "P", "集中载荷 P (N)", required=False, min=0)
    q_Nmm = parse_float(form, "q", "均布载荷 q (N/mm)", required=False, min=0)
    if P == 0 and q_Nmm == 0:
        raise FormError("请至少输入一种载荷：集中载荷 P 或均布载荷 q（都为 0 时梁上无载荷可算）")

    # 载荷位置 a：只在有集中载荷时才需要。必须在 Web 层用 mm 预检范围——
    # 公式库的报错是按 SI（m）写的，直接透传会把 "a=2.5, L=2" 这种 m 值甩给用户，看不懂。
    a_mm = None
    if P > 0 and (form.get("a") or "").strip():
        a_mm = parse_float(form, "a", "集中载荷位置 a (mm)", min=0, max=L_mm)

    # 弹性模量：下拉选材料（教材典型值）或自定义 GPa
    material = parse_choice(
        form, "material", "材料", tuple(ELASTIC_MODULUS) + ("custom",)
    )
    if material == "custom":
        E_GPa = parse_float(form, "E_gpa", "弹性模量 E (GPa)", min=0, exclusive_min=True)
        E_Pa = E_GPa * 1e9                     # GPa → Pa
        E_label = f"{E_GPa:g} GPa（自定义）"
    else:
        E_Pa = ELASTIC_MODULUS[material]
        E_label = f"{material}（教材典型值 E ≈ {E_Pa / 1e9:g} GPa）"

    # 截面惯性矩：矩形 / 实心圆（用库里的惯性矩函数，输入换算成 m）/ 自定义 mm⁴
    section = parse_choice(form, "section", "截面类型", ("rect", "circle", "custom"))
    if section == "rect":
        b_mm = parse_float(form, "b", "截面宽 b (mm)", min=0, exclusive_min=True)
        h_mm = parse_float(form, "h", "截面高 h (mm)", min=0, exclusive_min=True)
        I_m4 = beam_lib.rectangular_inertia(b_mm / 1000.0, h_mm / 1000.0)
        I_label = f"{b_mm:g} × {h_mm:g} 矩形（I = b·h³/12）"
    elif section == "circle":
        d_mm = parse_float(form, "d", "截面直径 d (mm)", min=0, exclusive_min=True)
        I_m4 = beam_lib.solid_circle_inertia(d_mm / 1000.0)
        I_label = f"实心圆 φ{d_mm:g}（I = π·d⁴/64）"
    else:
        I_mm4 = parse_float(form, "I", "惯性矩 I (mm⁴)", min=0, exclusive_min=True)
        I_m4 = I_mm4 * 1e-12                   # 1 mm⁴ = 1e-12 m⁴
        I_label = f"{I_mm4:g} mm⁴（自定义）"

    # ---- 输入层 → SI，唯一允许发生单位换算的地方 ----
    L_m = L_mm / 1000.0                        # mm → m
    q_Nm = q_Nmm * 1000.0                      # N/mm → N/m
    a_m = a_mm / 1000.0 if a_mm is not None else None

    if support == "simply":
        result = beam_lib.simply_supported(L=L_m, E=E_Pa, I=I_m4, P=P, a=a_m, q=q_Nm)
    else:
        result = beam_lib.cantilever(L=L_m, E=E_Pa, I=I_m4, P=P, a=a_m, q=q_Nm)

    # ---- 结果层：把库里按 m 给的位置量换回 mm，并回显换算链（教学 + 自查）----
    view = dict(result)                        # 拷贝一份，绝不改库返回的 dict
    view["max_deflection_at_x_mm"] = result["max_deflection_at_x_m"] * 1000.0
    view["max_moment_at_x_mm"] = result["max_moment_at_x_m"] * 1000.0
    view["closed_form_mm"] = (
        None if result["max_deflection_closed_form_m"] is None
        else result["max_deflection_closed_form_m"] * 1000.0
    )
    si_echo = [
        ("跨度 L", f"{L_mm:g} mm", f"{L_m:g} m"),
        ("弹性模量 E", E_label, f"{sig(E_Pa)} Pa"),
        ("截面惯性矩 I", I_label, f"{sig(I_m4)} m⁴"),
    ]
    if P > 0:
        si_echo.append(("集中载荷 P", f"{P:g} N", f"{P:g} N"))
    if a_mm is not None:
        si_echo.append(("载荷位置 a", f"{a_mm:g} mm", f"{a_m:g} m"))
    if q_Nmm > 0:
        si_echo.append(("均布载荷 q", f"{q_Nmm:g} N/mm", f"{q_Nm:g} N/m"))
    view["si_echo"] = si_echo
    return view


# --------------------------------------------------------------------------
# 螺栓组计算器：表单 → 公式库（单位与库一致：N、N·mm、mm，无需换算）
# --------------------------------------------------------------------------
def bolt_from_form(form) -> dict:
    case = parse_choice(
        form, "case", "工况",
        ("axial", "overturn", "flange", "torque", "preload"),
    )

    if case == "axial":
        F = parse_float(form, "F", "总轴向载荷 F (N)", min=0, exclusive_min=True)
        z = parse_int(form, "z", "螺栓个数 z", min=1)
        return bolt_lib.axial_uniform(F=F, z=z)

    if case == "overturn":
        M = parse_float(form, "M", "倾覆力矩 M (N·mm)", min=0)
        F_axial = parse_float(
            form, "F_axial", "叠加轴向载荷 F_axial (N)", required=False, min=0
        )
        if M == 0 and F_axial == 0:
            raise FormError("请输入倾覆力矩 M 或轴向载荷 F_axial（两者都为 0 时螺栓无载荷可算）")
        distances = parse_float_list(
            form, "distances", "各螺栓到翻转轴线的距离 (mm)"
        )
        return bolt_lib.overturning_about_axis(
            M=M, distances=distances, F_axial=F_axial
        )

    if case == "flange":
        M = parse_float(form, "M", "倾覆力矩 M (N·mm)", min=0)
        z = parse_int(form, "z", "螺栓个数 z（圆周均布至少 3 个）", min=3)
        D = parse_float(form, "D", "螺栓中心圆直径 D (mm)", min=0, exclusive_min=True)
        F_axial = parse_float(
            form, "F_axial", "叠加轴向载荷 F_axial (N)", required=False, min=0
        )
        if M == 0 and F_axial == 0:
            raise FormError("请输入倾覆力矩 M 或轴向载荷 F_axial（两者都为 0 时螺栓无载荷可算）")
        return bolt_lib.overturning_circular_flange(M=M, z=z, D=D, F_axial=F_axial)

    if case == "torque":
        T = parse_float(form, "T", "扭矩 T (N·mm)", min=0, exclusive_min=True)
        z = parse_int(form, "z", "螺栓个数 z", min=1)
        D = parse_float(form, "D", "螺栓中心圆直径 D (mm)", min=0, exclusive_min=True)
        return bolt_lib.torque_circular(T=T, z=z, D=D)

    # preload：普通螺栓靠摩擦承受横向载荷，求所需预紧力
    F_transverse = parse_float(form, "F_transverse", "横向总载荷 F_Σ (N)", min=0, exclusive_min=True)
    z = parse_int(form, "z", "螺栓个数 z", min=1)
    friction_coef = parse_float(form, "f", "摩擦系数 f", min=0, exclusive_min=True)
    joints = parse_int(form, "joints", "接合面对数 m", min=1)
    safety_factor = parse_float(form, "K_f", "可靠性系数 K_f", min=0, exclusive_min=True)
    return bolt_lib.transverse_friction_preload(
        F_transverse=F_transverse, z=z, friction_coef=friction_coef,
        joints=joints, safety_factor=safety_factor,
    )


# --------------------------------------------------------------------------
# 轴承寿命计算器：表单 → 公式库（N、r/min，与库一致）
# --------------------------------------------------------------------------
def bearing_from_form(form) -> dict:
    Fr = parse_float(form, "Fr", "径向载荷 Fr (N)", min=0)
    Fa = parse_float(form, "Fa", "轴向载荷 Fa (N)", min=0)
    if Fr == 0 and Fa == 0:
        raise FormError("径向与轴向载荷不能同时为 0（轴承上没有载荷就谈不上寿命）")
    Cr = parse_float(form, "Cr", "基本额定动载荷 Cr (N)", min=0, exclusive_min=True)
    Cor = parse_float(form, "Cor", "基本额定静载荷 C0r (N)", min=0, exclusive_min=True)
    n = parse_float(form, "n", "转速 n (r/min)", min=0, exclusive_min=True)
    f_t = parse_float(form, "f_t", "温度系数 f_t", required=False, default=1.0,
                      min=0, exclusive_min=True)
    f_p = parse_float(form, "f_p", "载荷系数 f_p", required=False, default=1.0,
                      min=0, exclusive_min=True)
    result = bearing_lib.deep_groove_ball_life(
        Fr=Fr, Fa=Fa, Cr=Cr, Cor=Cor, n=n,
        load_factor=f_p, temperature_factor=f_t,
    )
    # 纯轴向载荷时库返回 F_a/F_r = inf（float('inf')），页面上显示 ∞ 更友好。
    # 包装一份新 dict，不动库的返回值。
    view = dict(result)
    if view["Fa_over_Fr"] == float("inf"):
        view["Fa_over_Fr"] = None
    return view


# --------------------------------------------------------------------------
# 公差配合计算器：表单 → 公式库（mm、μm，与库一致）
# --------------------------------------------------------------------------
_CODE_RE = re.compile(r"[A-Za-z]{1,2}[0-9]{1,2}")   # H7 / g6 / js7 这种格式


def fits_deviation_from_form(form) -> dict:
    """单公差带偏差查询：φ30 H7 → 孔的 ES/EI。"""
    kind = parse_choice(form, "kind", "查孔还是轴", ("hole", "shaft"))
    nominal = parse_float(form, "nominal", "公称尺寸 (mm)",
                          min=0, exclusive_min=True, max=500)
    letter = (form.get("letter") or "").strip()
    if not re.fullmatch(r"[A-Za-z]{1,2}", letter):
        raise FormError(f"「基本偏差代号」请输入 1~2 个字母（如 H、g、js），当前输入：{letter!r}")
    grade = parse_int(form, "grade", "公差等级 IT", min=5, max=18)

    if kind == "hole":
        return fits_lib.hole_deviation(nominal, letter, grade)
    return fits_lib.shaft_deviation(nominal, letter, grade)


def fits_fit_from_form(form) -> dict:
    """配合查询：φ30 H7/g6 → 极限间隙 / 过盈、配合种类。"""
    nominal = parse_float(form, "nominal", "公称尺寸 (mm)",
                          min=0, exclusive_min=True, max=500)
    hole_code = (form.get("hole_code") or "").strip()
    shaft_code = (form.get("shaft_code") or "").strip()
    for label, code in (("孔公差带", hole_code), ("轴公差带", shaft_code)):
        if not _CODE_RE.fullmatch(code):
            raise FormError(
                f"「{label}」格式应为 字母+等级数字，如 H7、g6、js7（当前输入：{code!r}）"
            )
    return fits_lib.fit(nominal, hole_code, shaft_code)


# --------------------------------------------------------------------------
# 应用工厂与路由
# --------------------------------------------------------------------------
def create_app() -> Flask:
    app = Flask(__name__)
    app.jinja_env.filters["sig"] = sig

    # 材料下拉框的数据（名称, 显示文案），GPa 数值由库里的 SI 值换算显示
    materials = [
        (name, f"{name}（E ≈ {E / 1e9:g} GPa，教材典型值）")
        for name, E in ELASTIC_MODULUS.items()
    ]
    it_grades = list(range(5, 19))             # 本工具覆盖 IT5~IT18，与库一致

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/beam", methods=["GET", "POST"])
    def beam_page():
        result, errors = None, []
        # 只有 ValueError（FormError 是它的子类 + 公式库的输入报错）才转成
        # 友好提示；库的内部自检 AssertionError 不吞——那是 bug，不该伪装成输入错误。
        if request.method == "POST":
            try:
                result = beam_from_form(request.form)
                # 阶段 3：挠度 / 弯矩分布图。曲线数据直接取库返回的 curve（SI），
                # 换算与绘制在 plots.py；图与表格里的极值点互相印证。
                result["plot_uri"] = png_data_uri(beam_curves_png(
                    result["curve"], result["support"],
                    max_deflection_mm=result["max_deflection_mm"],
                    max_deflection_x_mm=result["max_deflection_at_x_mm"],
                    max_moment_Nm=result["max_moment_Nm"],
                    max_moment_x_mm=result["max_moment_at_x_mm"],
                ))
            except ValueError as exc:
                errors.append(str(exc))
        return render_template(
            "beam.html", result=result, errors=errors,
            form=request.form, materials=materials,
        )

    @app.route("/bolt-group", methods=["GET", "POST"])
    def bolt_group_page():
        result, errors, result_case = None, [], None
        if request.method == "POST":
            try:
                # 先看提交的是哪个工况的小表单（每个表单都带隐藏的 case 字段）
                result_case = request.form.get("case")
                result = bolt_from_form(request.form)
            except ValueError as exc:
                errors.append(str(exc))
        return render_template(
            "bolt_group.html", result=result, result_case=result_case,
            errors=errors, form=request.form,
        )

    @app.route("/bearing", methods=["GET", "POST"])
    def bearing_page():
        result, errors = None, []
        if request.method == "POST":
            try:
                result = bearing_from_form(request.form)
            except ValueError as exc:
                errors.append(str(exc))
        return render_template(
            "bearing.html", result=result, errors=errors, form=request.form,
        )

    @app.route("/fits", methods=["GET", "POST"])
    def fits_page():
        result, errors, result_mode = None, [], None
        if request.method == "POST":
            try:
                result_mode = request.form.get("mode")   # deviation / fit
                if result_mode == "fit":
                    result = fits_fit_from_form(request.form)
                else:
                    result_mode = "deviation"
                    result = fits_deviation_from_form(request.form)
            except ValueError as exc:
                errors.append(str(exc))
        return render_template(
            "fits.html", result=result, result_mode=result_mode,
            errors=errors, form=request.form, it_grades=it_grades,
        )

    return app


if __name__ == "__main__":
    # 本地开发入口：python app.py 后访问 http://127.0.0.1:5000
    # 部署（阶段 4）时会换 gunicorn，不依赖这里的 app.run。
    create_app().run(
        host="127.0.0.1",
        port=int(os.environ.get("PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG") == "1",
    )
