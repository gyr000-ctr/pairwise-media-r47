"""imgglow 核心实现：阈值高光提取、可分离高斯模糊与加法钳制合成。

仅使用 Python 标准库。校验后使用嵌套行/扁平索引进行计算，水平、垂直两遍
均为一维高斯卷积；默认边界复制边缘像素（clamp/replicate），因此核归一化时
常量区域不会产生暗边。
"""

from __future__ import annotations

import math

__all__ = [
    "Glow",
    "GlowError",
    "GlowArgumentError",
    "InvalidThresholdError",
    "InvalidSigmaError",
    "InvalidStrengthError",
    "InvalidImageError",
    "NonFiniteInputError",
    "gaussian_kernel",
    "kernel",
    "threshold_map",
    "bright_pass",
    "extract_bright",
    "gaussian_blur",
    "blur",
    "separable_gaussian",
    "gaussian",
    "gaussian_filter",
    "separable_convolve",
    "convolve_separable",
    "compose",
    "composite",
    "add_glow",
    "glow",
]


class GlowError(ValueError):
    """所有 imgglow 参数/输入错误的基类，可按 ``ValueError`` 捕获。"""


class GlowArgumentError(TypeError, GlowError):
    """参数类型错误；同时是 ``TypeError`` 与 ``ValueError``。"""


class InvalidThresholdError(GlowError):
    """threshold 不在 [0, 1] 或不是有限实数。"""


class InvalidSigmaError(GlowError):
    """sigma 不是有限正实数。"""


class InvalidStrengthError(GlowError):
    """strength 不是有限非负实数。"""


class InvalidImageError(GlowError):
    """图像为空、形状不合法或包含非数值元素。"""


class NonFiniteInputError(GlowError):
    """图像中包含 NaN 或无穷值。"""


def _real_float(value, exc_type, message):
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int):
        return float(value)
    if isinstance(value, float):
        fv = value
    else:
        raise GlowArgumentError(message + "，得到类型 %r" % type(value).__name__)
    if not math.isfinite(fv):
        raise exc_type(message + "，必须有限")
    return fv


def _validate_threshold(threshold):
    value = _real_float(
        threshold, InvalidThresholdError, "threshold 必须是 [0, 1] 内的有限实数"
    )
    if value < 0.0 or value > 1.0:
        raise InvalidThresholdError("threshold 必须在 [0, 1] 内，得到 %r" % (threshold,))
    return value


def _validate_sigma(sigma):
    value = _real_float(sigma, InvalidSigmaError, "sigma 必须是有限正实数")
    if value <= 0.0:
        raise InvalidSigmaError("sigma 必须严格大于 0，得到 %r" % (sigma,))
    return value


def _validate_strength(strength):
    value = _real_float(strength, InvalidStrengthError, "strength 必须是有限非负实数")
    if value < 0.0:
        raise InvalidStrengthError("strength 不能为负，得到 %r" % (strength,))
    return value


def _validate_radius(radius, sigma=None):
    if radius is None:
        scaled = 3.0 * sigma
        if not math.isfinite(scaled):
            raise InvalidSigmaError("sigma 过大，无法确定高斯半径")
        return max(1, int(math.ceil(scaled)))

    value = _real_float(radius, GlowArgumentError, "radius 必须是正整数")
    if value != math.floor(value):
        raise GlowArgumentError("radius 必须是整数，得到 %r" % (radius,))
    ivalue = int(value)
    if ivalue < 1:
        raise GlowError("radius 必须 >= 1，得到 %r" % (radius,))
    if ivalue > 1_000_000:
        raise GlowError("radius 过大：%r" % (radius,))
    return ivalue


def gaussian_kernel(sigma=1.0, radius=None):
    """返回对称、归一化的一维高斯核（list[float]）。"""
    sigma = _validate_sigma(sigma)
    radius = _validate_radius(radius, sigma)

    denom = 2.0 * sigma * sigma
    if denom == 0.0:  # 校验已保证 sigma > 0，这里仅作防御。
        half = [1.0] + [0.0] * radius
    else:
        half = [math.exp(-(i * i) / denom) for i in range(radius + 1)]

    # 对称核：total = half[0] + 2 * sum(half[1:])。
    total = half[0] + 2.0 * math.fsum(half[1:])
    if not math.isfinite(total) or total <= 0.0:
        raise InvalidSigmaError("无法为 sigma=%r 构造有限归一化核" % (sigma,))

    inv = 1.0 / total
    half = [v * inv for v in half]
    kernel = half[::-1] + half[1:]

    # 把 fsum 口径下的微小归一化残差补到中心权重，保证 sum(k) 尽量接近 1。
    residual = 1.0 - math.fsum(kernel)
    if residual:
        center = len(kernel) // 2
        kernel[center] += residual
    return kernel


# 常见函数式命名别名。
kernel = gaussian_kernel


def _finite_pixel(value):
    """热路径外使用的像素转换/校验。"""
    cls = type(value)
    if cls is float:
        fv = value
    elif cls is int or cls is bool:
        fv = float(value)
    elif isinstance(value, (int, float)):
        fv = float(value)
    else:
        raise GlowArgumentError("图像像素必须是实数，得到 %r" % type(value).__name__)
    if not math.isfinite(fv):
        raise NonFiniteInputError("图像包含 NaN 或无穷值")
    return fv


def _validate_image(image):
    """校验二维数值图像，返回 ``(flat_values, width, height)``。"""
    if not isinstance(image, (list, tuple)):
        raise InvalidImageError(
            "image 必须是二维 list/tuple，得到 %r" % type(image).__name__
        )
    if len(image) == 0:
        raise InvalidImageError("image 不能为空")

    flat = []
    width = None
    finite = math.isfinite
    append = flat.append

    for row_index, row in enumerate(image):
        if not isinstance(row, (list, tuple)):
            raise InvalidImageError(
                "image 的每一行必须是 list/tuple；第 %d 行类型为 %r"
                % (row_index, type(row).__name__)
            )
        row_width = len(row)
        if row_width == 0:
            raise InvalidImageError("image 行不能为空（第 %d 行）" % row_index)
        if width is None:
            width = row_width
        elif row_width != width:
            raise InvalidImageError(
                "image 必须是矩形：期望宽度 %d，第 %d 行宽度 %d"
                % (width, row_index, row_width)
            )

        for value in row:
            cls = type(value)
            if cls is float:
                fv = value
            elif cls is int or cls is bool:
                fv = float(value)
            elif isinstance(value, (int, float)):
                fv = float(value)
            else:
                raise GlowArgumentError(
                    "图像像素必须是实数，得到 %r" % type(value).__name__
                )
            if not finite(fv):
                raise NonFiniteInputError("图像包含 NaN 或无穷值")
            append(fv)

    if width is None or not flat:
        raise InvalidImageError("image 不能为空")
    return flat, width, len(image)


def _threshold_rows(image, threshold):
    """校验图像并直接构造高光行，减少一次百万级中间列表遍历。"""
    if not isinstance(image, (list, tuple)):
        raise InvalidImageError(
            "image 必须是二维 list/tuple，得到 %r" % type(image).__name__
        )
    if len(image) == 0:
        raise InvalidImageError("image 不能为空")

    bright_rows = []
    width = None
    has_bright = False
    finite = math.isfinite

    for row_index, row in enumerate(image):
        if not isinstance(row, (list, tuple)):
            raise InvalidImageError(
                "image 的每一行必须是 list/tuple；第 %d 行类型为 %r"
                % (row_index, type(row).__name__)
            )
        row_width = len(row)
        if row_width == 0:
            raise InvalidImageError("image 行不能为空（第 %d 行）" % row_index)
        if width is None:
            width = row_width
        elif row_width != width:
            raise InvalidImageError(
                "image 必须是矩形：期望宽度 %d，第 %d 行宽度 %d"
                % (width, row_index, row_width)
            )

        bright_row = []
        append = bright_row.append
        for value in row:
            cls = type(value)
            if cls is float:
                fv = value
            elif cls is int or cls is bool:
                fv = float(value)
            elif isinstance(value, (int, float)):
                fv = float(value)
            else:
                raise GlowArgumentError(
                    "图像像素必须是实数，得到 %r" % type(value).__name__
                )
            if not finite(fv):
                raise NonFiniteInputError("图像包含 NaN 或无穷值")
            if fv > threshold:
                append(fv)
                has_bright = True
            else:
                append(0.0)
        bright_rows.append(bright_row)

    return bright_rows, has_bright, width, len(image)


def _rows_from_flat(flat, width):
    return [flat[i : i + width] for i in range(0, len(flat), width)]


def _threshold_flat(flat, threshold):
    out = [v if v > threshold else 0.0 for v in flat]
    has_bright = any(v != 0.0 for v in out)
    return out, has_bright


def threshold_map(image, threshold=0.5):
    """函数式阈值提取：高于 threshold 的像素保留原值，其余为精确 0.0。"""
    th = _validate_threshold(threshold)
    flat, width, _ = _validate_image(image)
    bright, _ = _threshold_flat(flat, th)
    return _rows_from_flat(bright, width)


bright_pass = threshold_map
extract_bright = threshold_map


def _boundary_index(i, n, mode):
    if mode in ("edge", "replicate", "clamp", "nearest"):
        if i < 0:
            return 0
        if i >= n:
            return n - 1
        return i
    if mode in ("zero", "constant", "fixed"):
        return i if 0 <= i < n else -1
    if mode in ("wrap", "circular"):
        return i % n
    if mode in ("reflect", "mirror"):
        # 不重复边缘像素的反射（与 numpy reflect 一致的常用周期 2n-2）。
        if n == 1:
            return 0
        period = 2 * (n - 1)
        i %= period
        if i < 0:
            i += period
        return i if i < n else period - i
    if mode in ("symmetric",):
        # 重复边缘像素的镜像（numpy symmetric 口径）。
        if n == 1:
            return 0
        period = 2 * n
        i %= period
        if i < 0:
            i += period
        return i if i < n else period - 1 - i
    raise GlowArgumentError("未知边界模式 %r" % (mode,))


def _padded_row(row, radius, mode):
    n = len(row)
    if mode in ("edge", "replicate", "clamp", "nearest"):
        middle = row if isinstance(row, list) else list(row)
        return [row[0]] * radius + middle + [row[-1]] * radius
    if mode in ("zero", "constant", "fixed"):
        return [0.0] * radius + list(row) + [0.0] * radius
    return [row[_boundary_index(i, n, mode)] for i in range(-radius, n + radius)]


def _horizontal_blur(rows, kernel, radius, mode):
    width = len(rows[0]) if rows else 0
    symmetric = all(
        kernel[radius - i] == kernel[radius + i] for i in range(1, radius + 1)
    )
    blurred = []

    if symmetric:
        center_weight = kernel[radius]
        pairs = [(i, kernel[radius - i]) for i in range(1, radius + 1)]
        for row in rows:
            width = len(row)
            padded = _padded_row(row, radius, mode)
            acc = [center_weight * v for v in row]
            for i, weight in pairs:
                left = padded[radius - i : radius - i + width]
                right = padded[radius + i : radius + i + width]
                acc = [
                    a + weight * (lv + rv)
                    for a, lv, rv in zip(acc, left, right)
                ]
            blurred.append(acc)
        return blurred

    for row in rows:
        width = len(row)
        padded = _padded_row(row, radius, mode)
        acc = [0.0] * width
        for offset, weight in enumerate(kernel, start=-radius):
            source = padded[radius + offset : radius + offset + width]
            acc = [a + weight * v for a, v in zip(acc, source)]
        blurred.append(acc)
    return blurred


def _vertical_blur_rows(rows, kernel, radius, mode):
    height = len(rows)
    width = len(rows[0])
    symmetric = all(
        kernel[radius - i] == kernel[radius + i] for i in range(1, radius + 1)
    )

    if mode in ("edge", "replicate", "clamp", "nearest"):
        padded_rows = (
            [rows[0]] * radius + list(rows) + [rows[-1]] * radius
        )
    elif mode in ("zero", "constant", "fixed"):
        zero_row = [0.0] * width
        padded_rows = [zero_row] * radius + list(rows) + [zero_row] * radius
    else:
        padded_rows = [
            rows[_boundary_index(y, height, mode)]
            for y in range(-radius, height + radius)
        ]

    output = []
    if symmetric:
        center_weight = kernel[radius]
        pairs = [(i, kernel[radius - i]) for i in range(1, radius + 1)]
        for y in range(height):
            base_index = y + radius
            center_row = padded_rows[base_index]
            acc = [center_weight * v for v in center_row]
            for i, weight in pairs:
                up = padded_rows[base_index - i]
                down = padded_rows[base_index + i]
                acc = [
                    a + weight * (u + d)
                    for a, u, d in zip(acc, up, down)
                ]
            output.append(acc)
        return output

    for y in range(height):
        base_index = y + radius
        acc = [0.0] * width
        for offset, weight in enumerate(kernel, start=-radius):
            source = padded_rows[base_index + offset]
            acc = [a + weight * v for a, v in zip(acc, source)]
        output.append(acc)
    return output


def _vertical_blur_compose(rows, source, strength, kernel, radius, mode):
    """垂直卷积最后一对权重与加法合成融合，少一次百万级中间遍历。"""
    height = len(rows)
    width = len(rows[0])
    center_weight = kernel[radius]
    pairs = [(i, kernel[radius - i]) for i in range(1, radius + 1)]

    if mode in ("edge", "replicate", "clamp", "nearest"):
        padded_rows = (
            [rows[0]] * radius + list(rows) + [rows[-1]] * radius
        )
    elif mode in ("zero", "constant", "fixed"):
        zero_row = [0.0] * width
        padded_rows = [zero_row] * radius + list(rows) + [zero_row] * radius
    else:
        padded_rows = [
            rows[_boundary_index(y, height, mode)]
            for y in range(-radius, height + radius)
        ]

    # radius >= 1 时，把最后一对对称权重直接并入最终合成表达式。
    if pairs:
        last_i, last_weight = pairs[-1]
        prefix_pairs = pairs[:-1]
    else:
        last_i, last_weight = 0, 0.0
        prefix_pairs = []

    output = []
    for y in range(height):
        base_index = y + radius
        center_row = padded_rows[base_index]
        acc = [center_weight * v for v in center_row]
        for i, weight in prefix_pairs:
            up = padded_rows[base_index - i]
            down = padded_rows[base_index + i]
            acc = [
                a + weight * (u + d)
                for a, u, d in zip(acc, up, down)
            ]

        original_row = source[y]
        if pairs:
            up = padded_rows[base_index - last_i]
            down = padded_rows[base_index + last_i]
            if strength == 1.0:
                result = [
                    (qv if qv <= 1.0 else 1.0) if qv >= 0.0 else 0.0
                    for o, a, u, d in zip(original_row, acc, up, down)
                    for qv in [o + a + last_weight * (u + d)]
                ]
            else:
                result = [
                    (qv if qv <= 1.0 else 1.0) if qv >= 0.0 else 0.0
                    for o, a, u, d in zip(original_row, acc, up, down)
                    for qv in [
                        o
                        + strength
                        * (a + last_weight * (u + d))
                    ]
                ]
        else:
            if strength == 1.0:
                result = [
                    (qv if qv <= 1.0 else 1.0) if qv >= 0.0 else 0.0
                    for o, v in zip(original_row, acc)
                    for qv in [o + v]
                ]
            else:
                result = [
                    (qv if qv <= 1.0 else 1.0) if qv >= 0.0 else 0.0
                    for o, v in zip(original_row, acc)
                    for qv in [o + strength * v]
                ]
        output.append(result)
    return output


def gaussian_blur(image, sigma=1.0, radius=None, *, mode="edge", kernel=None):
    """对二维数值图像执行可分离高斯模糊，返回 ``list[list[float]]``。"""
    # 兼容 ``blur(image, kernel_sequence)`` 形式。
    if kernel is None and isinstance(sigma, (list, tuple)):
        kernel = sigma
        sigma = 1.0

    flat, width, height = _validate_image(image)
    rows = _rows_from_flat(flat, width)

    if kernel is None:
        kern = gaussian_kernel(sigma, radius)
    else:
        if not isinstance(kernel, (list, tuple)) or len(kernel) == 0:
            raise GlowArgumentError("kernel 必须是非空一维序列")
        try:
            kern = [float(v) for v in kernel]
        except (TypeError, ValueError) as exc:
            raise GlowArgumentError("kernel 元素必须是实数") from exc
        if not all(math.isfinite(v) for v in kern):
            raise GlowArgumentError("kernel 必须有限")
        if len(kern) % 2 == 0:
            raise GlowArgumentError("一维卷积核长度应为奇数")

    rad = (len(kern) - 1) // 2
    horizontal = _horizontal_blur(rows, kern, rad, mode)
    return _vertical_blur_rows(horizontal, kern, rad, mode)


# 常见命名别名。
blur = gaussian_blur
separable_gaussian = gaussian_blur
gaussian = gaussian_blur
gaussian_filter = gaussian_blur


def separable_convolve(image, kernel, mode="edge"):
    """用同一个一维核执行水平、垂直两遍卷积。"""
    return gaussian_blur(image, kernel=kernel, mode=mode)


convolve_separable = separable_convolve


def _clamp01(value):
    if value <= 1.0:
        return 0.0 if value < 0.0 else value
    return 1.0


def compose(original, glow_layer, strength=1.0):
    """加法合成后钳制到 [0, 1]。"""
    op, w, h = _validate_image(original)
    gp, gw, gh = _validate_image(glow_layer)
    if (w, h) != (gw, gh) or len(gp) != len(op):
        raise InvalidImageError("original 与 glow_layer 的形状必须一致")
    s = _validate_strength(strength)
    out = []
    for y in range(h):
        start = y * w
        orow = op[start : start + w]
        grow = gp[start : start + w]
        if s == 1.0:
            out.append([_clamp01(o + g) for o, g in zip(orow, grow)])
        elif s == 0.0:
            out.append(list(orow))
        else:
            out.append([_clamp01(o + s * g) for o, g in zip(orow, grow)])
    return out


composite = compose
add_glow = compose


class Glow:
    """阈值高光 + 可分离高斯模糊 + 加法钳制的发光效果。"""

    def __init__(self, threshold=0.5, sigma=1.0, radius=None, *, mode="edge"):
        self.threshold = _validate_threshold(threshold)
        self.sigma = _validate_sigma(sigma)
        self.radius = _validate_radius(radius, self.sigma)
        if mode not in (
            "edge",
            "replicate",
            "clamp",
            "nearest",
            "zero",
            "constant",
            "fixed",
            "wrap",
            "circular",
            "reflect",
            "mirror",
            "symmetric",
        ):
            raise GlowArgumentError("未知边界模式 %r" % (mode,))
        self.mode = mode
        self._kernel = gaussian_kernel(self.sigma, self.radius)

    def kernel(self, sigma=None, radius=None):
        """返回一维高斯核；无参数时返回当前实例 sigma/radius 的核副本。"""
        if sigma is None and radius is None:
            return list(self._kernel)
        if sigma is None:
            sigma = self.sigma
        return gaussian_kernel(sigma, radius)

    gaussian_kernel = kernel
    make_kernel = kernel

    def threshold_map(self, image, threshold=None):
        """只保留高于阈值的原像素，其余位置为 0.0。"""
        th = self.threshold if threshold is None else _validate_threshold(threshold)
        flat, width, _ = _validate_image(image)
        bright, _ = _threshold_flat(flat, th)
        return _rows_from_flat(bright, width)

    # 语义别名，兼容可能的命名习惯。
    bright_pass = threshold_map
    extract_bright = threshold_map

    def blur(self, image, sigma=None, radius=None, *, mode=None, kernel=None, **kwargs):
        """使用当前高斯参数执行可分离模糊。"""
        if kernel is None and isinstance(sigma, (list, tuple)):
            kernel = sigma
            sigma = None
        border_mode = mode or kwargs.get("border") or kwargs.get("padding") or self.mode
        if sigma is None and radius is None and kernel is None:
            kern = self._kernel
        elif kernel is None:
            kern = self.kernel(sigma if sigma is not None else self.sigma, radius)
        else:
            kern = kernel
        return gaussian_blur(
            image,
            self.sigma if sigma is None else sigma,
            radius,
            mode=border_mode,
            kernel=kern,
        )

    gaussian_blur = blur
    separable_blur = blur
    separable_gaussian = blur
    gaussian = blur
    gaussian_filter = blur

    def convolve(self, image, kernel=None, mode=None, **kwargs):
        if kernel is None:
            raise GlowArgumentError("convolve 需要一维 kernel 参数")
        return self.blur(
            image,
            kernel=kernel,
            mode=mode,
            **kwargs,
        )

    separable_convolve = convolve

    def compose(self, original, glow_layer, strength=1.0):
        return compose(original, glow_layer, strength)

    composite = compose
    add_glow = compose

    def apply(
        self,
        image,
        strength=1.0,
        mode=None,
        *,
        border=None,
        padding=None,
        sigma=None,
        radius=None,
        **kwargs,
    ):
        """应用完整发光管线并返回新的嵌套 list 图像。"""
        s = _validate_strength(strength)
        border_mode = mode or border or padding or self.mode

        # 强度为 0 时严格恒等（先做有限性/形状校验，再做防御性复制）。
        if s == 0.0:
            flat, width, _ = _validate_image(image)
            return _rows_from_flat(flat, width)

        bright_rows, has_bright, width, _ = _threshold_rows(image, self.threshold)
        if not has_bright:
            # 已在 _threshold_rows 中完成有限性校验；这里只做 float 防御复制。
            return [[float(v) for v in row] for row in image]

        if sigma is None and radius is None:
            kern = self._kernel
        else:
            kern = self.kernel(
                self.sigma if sigma is None else sigma, radius
            )
        rad = (len(kern) - 1) // 2
        horizontal = _horizontal_blur(bright_rows, kern, rad, border_mode)
        del bright_rows
        return _vertical_blur_compose(
            horizontal, image, s, kern, rad, border_mode
        )

    # 常见近名 API 兼容。
    glow = apply
    render = apply
    process = apply


def glow(image, threshold=0.5, sigma=1.0, strength=1.0, radius=None, *, mode="edge"):
    """一次性函数式发光 API。"""
    return Glow(threshold, sigma, radius, mode=mode).apply(image, strength)
