"""Web 表单解析与校验（阶段 2）

【这层存在的理由】
公式库（calcs/）只认"干净的数字"，脏输入的拦截必须在进库之前完成：
  1. 用户在表单里输入的永远是**字符串**，"2000"、" 2000 "、"2e3"、甚至 "２０００"（全角）
     都要先转成 float，转不动就要给一句**人看得懂**的中文提示，
     而不是让 Python 的 ValueError 原样炸成 500 页面。
  2. 报错信息必须带上**字段名和单位**（如「跨度 L (mm)」），
     用户才能知道去改哪一格——这是表单校验和库内校验最大的区别。
  3. 库内的报错（ValueError，例如公差代号未内置）消息本身写得很好，
     Web 层原样展示即可，不重复造轮子。

【本文件不含任何业务公式】它只是"输入 → 数字"的翻译器，
翻译失败时抛 FormError（也是 ValueError 的子类，方便路由统一捕获）。
"""

from __future__ import annotations

import math
import re
from typing import Mapping, Optional


class FormError(ValueError):
    """表单输入不合法。消息直接展示给用户，所以必须是友好中文。"""


def _short(raw: str, limit: int = 24) -> str:
    """把用户的原始输入截短后放进报错信息，避免一行报错被撑爆。"""
    raw = raw.strip()
    return raw if len(raw) <= limit else raw[: limit - 1] + "…"


def parse_float(
    form: Mapping[str, str],
    field: str,
    label: str,
    *,
    required: bool = True,
    default: Optional[float] = None,
    min: Optional[float] = None,
    max: Optional[float] = None,
    exclusive_min: bool = False,
) -> float:
    """从表单取一个浮点数。

    form            : request.form（或测试里手工构造的 dict）
    field / label   : 字段英文名（表单 name）/ 中文名（报错时展示）
    required=False  : 留空时返回 default（用于"可选载荷"这类字段）
    min/max         : 允许的闭区间；exclusive_min=True 时下界改为开区间（必须 > min）
    """
    raw = (form.get(field) or "").strip()
    if not raw:
        if required:
            raise FormError(f"请填写「{label}」")
        return default if default is not None else 0.0

    # float() 能处理 "2e3"、全角数字等，但不能处理 "12abc"；
    # 全角空格先归一化，避免"看起来填了其实没填"的诡异报错。
    raw_normalized = raw.replace("\u3000", " ").replace("，", ",")
    try:
        value = float(raw_normalized)
    except ValueError:
        raise FormError(f"「{label}」请输入数字（当前输入：{_short(raw)}）") from None

    if not math.isfinite(value):
        raise FormError(f"「{label}」请输入有限的数字（当前输入：{_short(raw)}）")
    if min is not None:
        if exclusive_min and value <= min:
            raise FormError(f"「{label}」必须大于 {min:g}")
        if not exclusive_min and value < min:
            raise FormError(f"「{label}」不能小于 {min:g}")
    if max is not None and value > max:
        raise FormError(f"「{label}」不能大于 {max:g}")
    return value


def parse_int(
    form: Mapping[str, str],
    field: str,
    label: str,
    *,
    required: bool = True,
    default: Optional[int] = None,
    min: Optional[int] = None,
    max: Optional[int] = None,
) -> int:
    """从表单取一个整数（螺栓个数、公差等级这类字段）。

    "4.0" 这种写法宽容接受（用户在数字框里多打了个 .0 很常见），"4.5" 则拒绝。
    """
    raw = (form.get(field) or "").strip()
    if not raw:
        if required:
            raise FormError(f"请填写「{label}」")
        return default if default is not None else 0

    try:
        value = float(raw.replace("\u3000", " "))
    except ValueError:
        raise FormError(f"「{label}」请输入整数（当前输入：{_short(raw)}）") from None
    if value != int(value):
        raise FormError(f"「{label}」必须是整数（当前输入：{_short(raw)}）")
    value = int(value)

    if min is not None and value < min:
        raise FormError(f"「{label}」不能小于 {min}")
    if max is not None and value > max:
        raise FormError(f"「{label}」不能大于 {max}")
    return value


def parse_float_list(
    form: Mapping[str, str],
    field: str,
    label: str,
    *,
    min_count: int = 1,
) -> list:
    """从表单取一组浮点数（螺栓到翻转轴线的距离这种"列表"字段）。

    分隔符宽容处理：逗号（半角/全角）、分号、空格、换行都行——
    用户从 Excel 里粘一列数过来是最常见用法，粘贴出来就是换行分隔。
    """
    raw = (form.get(field) or "").strip()
    if not raw:
        raise FormError(f"请填写「{label}」")

    parts = re.split(r"[,，;；\s]+", raw.replace("\u3000", " "))
    values = []
    for part in parts:
        if not part:
            continue
        try:
            value = float(part)
        except ValueError:
            raise FormError(
                f"「{label}」里有不是数字的内容：{_short(part)}。"
                f"请用逗号或空格分隔，例如：200, 200, 50, 50"
            ) from None
        if not math.isfinite(value):
            raise FormError(f"「{label}」里出现了无效数值：{_short(part)}")
        values.append(value)

    if len(values) < min_count:
        raise FormError(f"「{label}」至少需要 {min_count} 个数（当前只解析出 {len(values)} 个）")
    return values


def parse_choice(form: Mapping[str, str], field: str, label: str, allowed) -> str:
    """从表单取一个枚举值。枚举不合法按"请重新选择"提示，而不是 500。"""
    raw = (form.get(field) or "").strip()
    if raw not in allowed:
        raise FormError(f"「{label}」请从下拉列表中选择（当前输入：{_short(raw)}）")
    return raw
