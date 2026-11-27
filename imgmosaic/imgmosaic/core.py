"""imgmosaic 核心 API（仅使用 Python 标准库）。

提供两种马赛克（像素化）模式：

1. 整数块尺寸 ``Mosaic().apply(image, block)``：``block`` 为正整数，每个块内
   的像素取算术平均（等价于块内和 / 块面积），图像边缘不足一整块的部分块
   同样按其实际像素数取均值。
2. 非均匀块网格 ``Mosaic().apply(image, (col_widths, row_heights))``：由列宽
   数组与行高数组定义块边界，块与像素单元按重叠面积加权平均。像素值视为
   覆盖在单位方格上的分段常数函数，因此该加权平均就是块上解析积分 / 块面积，
   块边界切过像素时同样成立。

输出始终为与输入同尺寸的 ``list[list[float]]``，且不修改输入图像。
"""

import math


class GridLengthError(ValueError):
    """非均匀块网格的长度/覆盖范围与图像尺寸不匹配。

    继承自 :class:`ValueError`，因此既可按类型与“非法块尺寸”区分，
    也仍可被 ``except ValueError`` 捕获。
    """


def _validate_image(image):
    """校验二维图像，返回 (高度, 宽度)。"""
    if not isinstance(image, list) or len(image) == 0:
        raise ValueError("image 必须为非空的二维列表")
    first = image[0]
    if not isinstance(first, (list, tuple)) or len(first) == 0:
        raise ValueError("image 必须为非空的二维列表")
    width = len(first)
    height = len(image)
    for row in image:
        if not isinstance(row, (list, tuple)) or len(row) != width:
            raise ValueError("image 各行必须等长且非空")
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("image 像素值必须为实数")
            if not math.isfinite(value):
                raise ValueError("image 像素值必须有限，不能为 NaN 或 Inf")
    return height, width


def _boundaries(widths, dim, label):
    """把宽度/高度数组转换为边界坐标，并校验其恰好覆盖 dim。"""
    if len(widths) == 0:
        raise GridLengthError("%s数组不能为空" % label)
    bounds = [0.0]
    total = 0.0
    for width in widths:
        if not isinstance(width, (int, float)) or isinstance(width, bool):
            raise ValueError("%s必须为实数数组" % label)
        if not math.isfinite(width):
            raise ValueError("%s必须为有限正数" % label)
        if width <= 0:
            raise ValueError("%s必须全部为正数" % label)
        total += float(width)
        bounds.append(total)
    if abs(total - dim) > 1e-9 * max(1, dim):
        raise GridLengthError(
            "%s之和 %.17g 与图像对应尺寸 %d 不符" % (label, total, dim)
        )
    # 端点吸附到精确边界，保证网格严格覆盖整幅图像。
    bounds[-1] = float(dim)
    return bounds


def _build_grid(block, height, width):
    """校验非均匀网格参数，返回 (x 边界, y 边界)。"""
    if len(block) != 2:
        raise GridLengthError(
            "非均匀块网格必须是 (列宽数组, 行高数组)，长度为 2，实际长度为 %d"
            % len(block)
        )
    col_widths, row_heights = block
    if not isinstance(col_widths, (list, tuple)) or not isinstance(
        row_heights, (list, tuple)
    ):
        raise ValueError("非均匀块网格的列宽/行高必须为数组")
    xs = _boundaries(col_widths, width, "列宽")
    ys = _boundaries(row_heights, height, "行高")
    return xs, ys


def _overlap_parts(bounds, dim):
    """每个块在某一轴上切到的单元：返回 [[(单元下标, 重叠长度), ...], ...]。"""
    parts = []
    for k in range(len(bounds) - 1):
        lo, hi = bounds[k], bounds[k + 1]
        index = int(math.floor(lo))
        if index < 0:
            index = 0
        current = []
        while index < dim and index < hi:
            cell_lo = lo if index < lo else float(index)
            cell_hi = hi if (index + 1) > hi else float(index + 1)
            current.append((index, cell_hi - cell_lo))
            index += 1
        parts.append(current)
    return parts


def _apply_grid(image, height, width, xs, ys):
    """非均匀块网格：面积加权的块均值，再按面积重采样回像素。

    第一步把像素值视为单位方格上的分段常数函数，块均值为块上解析积分
    除以块面积（块边界切过像素时按重叠面积加权）。第二步马赛克结果是
    块网格上的分段常数函数，输出像素为该函数在像素单元上的积分除以
    像素面积，即覆盖该像素的各块均值按重叠面积加权平均。整除时每个
    像素只属于一个块，结果与整数块路径完全相同。
    """
    col_parts = _overlap_parts(xs, width)
    row_parts = _overlap_parts(ys, height)
    accum = [[0.0] * width for _ in range(height)]
    coverage = [[0.0] * width for _ in range(height)]

    for rows in row_parts:
        for cols in col_parts:
            weighted_sum = 0.0
            weight_sum = 0.0
            for i, overlap_h in rows:
                row = image[i]
                for j, overlap_w in cols:
                    weight = overlap_h * overlap_w
                    weighted_sum += row[j] * weight
                    weight_sum += weight
            # weight_sum 即该块实际覆盖面积（分段常数下与块面积相等）。
            mean = weighted_sum / weight_sum
            for i, overlap_h in rows:
                accum_row = accum[i]
                cover_row = coverage[i]
                for j, overlap_w in cols:
                    weight = overlap_h * overlap_w
                    accum_row[j] += mean * weight
                    cover_row[j] += weight

    # 网格严格覆盖整幅图像，每个像素的覆盖面积为 1（允许浮点误差）。
    for i in range(height):
        accum_row = accum[i]
        cover_row = coverage[i]
        for j in range(width):
            accum_row[j] = accum_row[j] / cover_row[j]
    return accum


def _apply_uniform(image, height, width, block):
    """整数块：块内像素算术平均，边缘部分块按实际像素数取均值。"""
    out = [[0.0] * width for _ in range(height)]
    row_starts = range(0, height, block)
    col_starts = range(0, width, block)

    for r0 in row_starts:
        r1 = r0 + block
        if r1 > height:
            r1 = height
        for c0 in col_starts:
            c1 = c0 + block
            if c1 > width:
                c1 = width
            pixel_sum = 0.0
            for i in range(r0, r1):
                row = image[i]
                for j in range(c0, c1):
                    pixel_sum += row[j]
            # 块面积即像素数：均值 == 块内和 / 面积。
            mean = pixel_sum / ((r1 - r0) * (c1 - c0))
            for i in range(r0, r1):
                out_row = out[i]
                for j in range(c0, c1):
                    out_row[j] = mean
    return out


class Mosaic:
    """马赛克（像素化）处理器，纯 Python 标准库实现。"""

    def apply(self, image, block):
        """对图像做马赛克处理。

        :param image: 二维列表（行 × 列），元素为有限实数。
        :param block: 正整数表示均匀块尺寸；或二元组
            ``(列宽数组, 行高数组)`` 表示非均匀块网格。
        :return: 与 ``image`` 同尺寸的 ``list[list[float]]``，输入不被修改。
        :raises ValueError: 图像非法、块尺寸非正、像素含 NaN/Inf，
            或网格宽度/高度非正。
        :raises TypeError: 参数类型不正确。
        :raises GridLengthError: 非均匀网格长度/覆盖范围与图像不符。
        """
        height, width = _validate_image(image)

        if isinstance(block, (tuple, list)):
            xs, ys = _build_grid(block, height, width)
            return _apply_grid(image, height, width, xs, ys)

        if not isinstance(block, int):
            # 含浮点（即使是 2.0 这种整数值）、字符串等：块尺寸必须是整数。
            raise ValueError(
                "block 必须为正整数，或 (列宽数组, 行高数组) 形式的非均匀网格，"
                "收到 %r" % (block,)
            )
        if block <= 0:
            raise ValueError("block 块尺寸必须为正整数，收到 %r" % block)

        if block == 1:
            # 恒等：逐像素拷贝并统一转为 float。
            return [[float(value) for value in row] for row in image]

        return _apply_uniform(image, height, width, block)
