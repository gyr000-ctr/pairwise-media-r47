"""imgnormalize —— 直方图归一（全局分位拉伸 + 分块 CLAHE 式局部拉伸）。

设计要点
--------
* 输入/输出均为 ``[[v, ...], ...]`` 的嵌套 Python 列表（也接受元组等序列，
  以及一维列表），像素为浮点数。
* **全局模式**（默认）：用 ``lo`` / ``hi`` 两个分位点把灰度线性拉伸到
  ``[0, 1]``::

      y = clip((v - q_lo) / (q_hi - q_lo), 0, 1)

  映射严格单调不减（IEEE 舍入对减法与正数除法均保持序关系），且两个分位点
  对应的像素值精确落到 0 / 1。分位点采用线性插值（与 numpy ``percentile``
  的默认 "linear" 定义一致）。
* **局部模式**（CLAHE 式）：把图像切成 ``ny x nx`` 个均衡分块，每块独立求
  ``lo`` / ``hi`` 分位点并做同一线性拉伸；每个像素的最终值由包围它的
  至多 4 个块中心做双线性加权融合（权重只与像素坐标有关且随位置连续），
  因此块边界没有跳变（无接缝），输出仍落在 ``[0, 1]``。
* **常量保护**：块（或整幅图）的分位区间退化（``q_hi <= q_lo``）时退化为
  恒等映射，因此常量图逐像素原样输出（误差为 0，不放大任何噪声）；
  分块退化为「每像素一块」时同样直接返回原图。
* 只使用 Python 标准库；不修改输入；无全局可变状态（实例可并发共享）。
"""

from __future__ import annotations

import math
import numbers
from itertools import chain

__all__ = [
    "Normalize",
    "LocalNormalize",
    "GlobalNormalize",
    "NormalizeError",
    "NormalizeTypeError",
    "NormalizeShapeError",
    "NormalizeValueError",
    "NonFinitePixelError",
    "NaNPixelError",
    "InfinitePixelError",
    "InvalidQuantileError",
    "InvalidParameterError",
    "normalize",
    "stretch",
    "quantile",
    "quantiles",
    "percentile",
    "apply_local",
    "clahe",
    "Normalizer",
    "QuantileStretch",
    "CLAHE",
    "Local",
    "Global",
    "stretch_quantile",
    "global_stretch",
    "local_stretch",
    "local_normalize",
]

# 允许作为「块网格（每轴块数）」别名的关键字。
_GRID_KEYS = (
    "blocks",
    "tiles",
    "grid",
    "tile_grid",
    "grid_size",
    "grid_shape",
    "n_blocks",
    "num_blocks",
    "ntiles",
    "n_tiles",
    "block",
    "block_grid",
    "blocks_xy",
    "tiles_xy",
    "block_count",
    "tile_count",
    "block_grid_size",
    "tile_grid_size",
    "tile_counts",
    "block_counts",
    "blocks_per_axis",
    "tiles_per_axis",
    "tile",
    "nb",
)
# 允许作为「块边长（像素）」别名的关键字（整数或 (h, w) 二元组）。
_SIZE_KEYS = (
    "block_size",
    "tile_size",
    "cell_size",
    "block_px",
    "tile_px",
    "block_shape",
    "tile_shape",
    "chunk",
    "block_width",
    "tile_width",
    "block_hw",
    "tile_hw",
    "cell",
)
# 允许作为「是否走局部模式」别名的关键字。
_MODE_KEYS = (
    "local",
    "mode",
    "method",
    "kind",
    "local_mode",
    "local_stretch",
    "tiled",
    "clahe",
)


# ---------------------------------------------------------------------------
# 异常：非法输入按类别可区分，且都是 ValueError（兼容既有调用方）
# ---------------------------------------------------------------------------
class NormalizeError(ValueError):
    """所有 imgnormalize 错误的基类（同时是 ValueError）。"""


class NormalizeTypeError(NormalizeError, TypeError):
    """类型不合法（非数值像素、非序列输入等）。"""


class NormalizeShapeError(NormalizeError):
    """形状不合法（空图像、空行、参差不齐等）。"""


class NormalizeValueError(NormalizeError):
    """数值或参数不合法（NaN/Inf、越界分位、lo/hi 非法等）。"""


class NonFinitePixelError(NormalizeValueError):
    """图像像素含 NaN 或无穷（可按类型区分）。"""


class NaNPixelError(NonFinitePixelError):
    """图像像素含 NaN。"""


class InfinitePixelError(NonFinitePixelError):
    """图像像素含 +Inf / -Inf。"""


class InvalidQuantileError(NormalizeValueError):
    """分位参数越界或非有限。"""


class InvalidParameterError(NormalizeValueError):
    """构造/调用参数非法（块网格、lo/hi、mode 等）。"""


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------
def _is_number(value):
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _check_quantile(q, name="q"):
    """校验分位参数，返回 float。越界/非有限抛 InvalidQuantileError。"""
    if not _is_number(q):
        raise NormalizeTypeError(
            f"{name} 必须是 int 或 float，实际为 {type(q).__name__}"
        )
    q = float(q)
    if not math.isfinite(q) or q < 0.0 or q > 1.0:
        raise InvalidQuantileError(
            f"{name} 必须位于 [0, 1] 且有限，实际为 {q!r}"
        )
    return q


def _coerce_image(image):
    """把输入规范成 ``(height, width, flat, was_1d)``。

    ``flat`` 为按行优先展平的 float 列表，且已校验全部有限；输入不被修改。
    """
    if image is None or isinstance(image, (str, bytes, bytearray)):
        raise NormalizeTypeError("image 必须是数值二维序列")
    try:
        rows = list(image)
    except TypeError:
        raise NormalizeTypeError(
            f"image 必须是可迭代的序列，实际为 {type(image).__name__}"
        ) from None
    if not rows:
        raise NormalizeShapeError("image 不能为空")

    first = rows[0]
    if _is_number(first):
        # 一维输入按单行处理，返回时再还原形状。
        was_1d = True
        try:
            flat = [float(v) for v in rows]
        except (TypeError, ValueError):
            raise NormalizeTypeError("image 含有无法转换为浮点数的元素") from None
        if not all(map(math.isfinite, flat)):
            _raise_non_finite(flat)
        return 1, len(rows), flat, True

    if isinstance(first, (str, bytes, bytearray)) or not hasattr(first, "__len__"):
        raise NormalizeTypeError("image 的每一行必须是数值序列")
    width = len(first)
    if width == 0:
        raise NormalizeShapeError("image 不能包含空行")

    for row in rows:
        tro = type(row)
        if tro is list or tro is tuple:
            # 常见情形走快路径：list/tuple 的长度与可迭代性无需再检查。
            if len(row) != width:
                raise NormalizeShapeError(
                    f"image 各行长度必须一致（期望 {width}，实际 {len(row)}）"
                )
            continue
        if isinstance(row, (str, bytes, bytearray)) or not hasattr(row, "__len__"):
            raise NormalizeTypeError("image 的每一行必须是数值序列")
        if len(row) != width:
            raise NormalizeShapeError(
                f"image 各行长度必须一致（期望 {width}，实际 {len(row)}）"
            )

    try:
        flat = [float(v) for v in chain.from_iterable(rows)]
    except (TypeError, ValueError):
        raise NormalizeTypeError("image 含有无法转换为浮点数的元素") from None

    if len(flat) != len(rows) * width:
        # 序列自报长度与实际迭代结果不符（如自定义容器）。
        raise NormalizeShapeError("image 的元素总数与形状不一致")
    if not all(map(math.isfinite, flat)):
        # 按类别区分 NaN 与 Inf（两者都是 NormalizeValueError 的子类）。
        for r, row in enumerate(rows):
            for c, v in enumerate(row):
                fv = float(v)
                if fv != fv:
                    raise NaNPixelError(f"image 在行 {r} 列 {c} 含 NaN")
                if math.isinf(fv):
                    raise InfinitePixelError(f"image 在行 {r} 列 {c} 含 Inf")
    return len(rows), width, flat, False


def _raise_non_finite(flat):
    for i, v in enumerate(flat):
        if v != v:
            raise NaNPixelError(f"image 在第 {i} 个像素含 NaN")
        if math.isinf(v):
            raise InfinitePixelError(f"image 在第 {i} 个像素含 Inf")


def _quantile_sorted(sorted_vals, q):
    """在已排序列表上求线性插值分位点（numpy "linear" 定义）。"""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = q * (n - 1)
    i = int(pos)
    if i >= n - 1:
        return sorted_vals[-1]
    frac = pos - i
    if frac == 0.0:
        return sorted_vals[i]
    lo = sorted_vals[i]
    hi = sorted_vals[i + 1]
    return lo + (hi - lo) * frac


def _edges(n, k):
    """把 ``[0, n)`` 均匀分成 k 段（余数摊到前面），返回 k+1 个边界。"""
    return [(n * i) // k for i in range(k + 1)]


def _axis_segments(n, k):
    """把逐坐标的块中心插值权重压缩成连续区段。

    返回 ``(start, stop, i0, i1, c0, span)`` 列表：对 ``start <= p < stop``，
    参与插值的块下标恒为 ``i0 / i1``，权重为 ``(p + 0.5 - c0) / span``。
    像素坐标取 ``r + 0.5``，块中心取块区间中点 ``(首+尾)/2``，两端夹取。
    区段化后不再需要长度为 ``n`` 的权重表（百万行输入也不分配大表）。
    """
    if k <= 1:
        return [(0, n, 0, 0, 0.0, 1.0)] if n else []
    edges = _edges(n, k)
    centers = [(edges[i] + edges[i + 1]) * 0.5 for i in range(k)]
    last = k - 1

    # 区段边界可以直接由块中心解析求出（O(k)），无需逐坐标扫描：
    # 像素坐标 p = r + 0.5；块 i 覆盖 p in [centers[i], centers[i+1])，
    # 即 r in [ceil(centers[i]-0.5), ceil(centers[i+1]-0.5))。
    def first_r_at_or_after(center):
        return int(math.ceil(center - 0.5))

    segments = []
    r0 = max(0, min(n, first_r_at_or_after(centers[0])))
    rl = max(r0, min(n, first_r_at_or_after(centers[last])))
    if r0 > 0:
        segments.append((0, r0, 0, 0, 0.0, 1.0))
    for i in range(last):
        start = max(r0, first_r_at_or_after(centers[i]))
        stop = min(rl, first_r_at_or_after(centers[i + 1]))
        if stop > start:
            segments.append((start, stop, i, i + 1, centers[i], centers[i + 1] - centers[i]))
    if rl < n:
        segments.append((rl, n, last, last, 0.0, 1.0))
    # 以全零跨度补空（仅在极端退化尺寸下出现）。
    if not segments:
        segments.append((0, n, 0, 0, 0.0, 1.0))
    return segments


def _parse_positive_int(value, what):
    if not _is_number(value):
        raise NormalizeTypeError(f"{what} 必须是正整数，实际为 {value!r}")
    fvalue = float(value)
    if not math.isfinite(fvalue) or fvalue <= 0.0 or fvalue != math.floor(fvalue):
        raise InvalidParameterError(f"{what} 必须是正整数，实际为 {value!r}")
    return int(fvalue)


def _parse_grid(value):
    """把块网格参数解析成 ``(ny, nx)``（未按图像尺寸夹取）。"""
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise InvalidParameterError(
                f"块网格必须是正整数或 (ny, nx) 二元组，实际为 {value!r}"
            )
        return (
            _parse_positive_int(value[0], "块网格行数"),
            _parse_positive_int(value[1], "块网格列数"),
        )
    if isinstance(value, dict):
        raise NormalizeTypeError("块网格参数不能是 dict")
    n = _parse_positive_int(value, "块数")
    return n, n


def _parse_size(value, what="块边长"):
    """把块边长参数解析成 ``(bh, bw)``（像素）。"""
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise InvalidParameterError(
                f"{what} 必须是正数或 (h, w) 二元组，实际为 {value!r}"
            )
        bh = _parse_positive_number(value[0], what)
        bw = _parse_positive_number(value[1], what)
        return bh, bw
    b = _parse_positive_number(value, what)
    return b, b


def _parse_positive_number(value, what):
    if not _is_number(value):
        raise NormalizeTypeError(f"{what} 必须是正数，实际为 {value!r}")
    fvalue = float(value)
    if not math.isfinite(fvalue) or fvalue <= 0.0:
        raise InvalidParameterError(f"{what} 必须是有限正数，实际为 {value!r}")
    return fvalue


def _grid_from_size(value, height, width):
    """把「块边长（像素）」转换成块网格数量。"""
    bh, bw = _parse_size(value)
    ny = max(1, int(math.ceil(height / bh)))
    nx = max(1, int(math.ceil(width / bw)))
    return min(ny, height), min(nx, width)


def _pop_option(options, keys):
    """按别名顺序取出一个选项。"""
    for key in keys:
        if key in options:
            return options.pop(key)
    return None


def _normalize_mode(mode):
    """把 mode / local 选项解释为 ``True`` / ``False``；``None`` 表示未指定。

    额外接受 ``1`` / ``0`` 与常见字符串（local / clahe / global ...），
    兼容不同命名习惯的调用方。
    """
    if mode is None:
        return None
    if isinstance(mode, bool):
        return mode
    if isinstance(mode, numbers.Real) and mode in (0, 1):
        return bool(mode)
    if isinstance(mode, str):
        m = mode.strip().lower()
        if m in ("local", "clahe", "tile", "tiled", "block", "blocks", "adaptive"):
            return True
        if m in ("global", "full", "whole", "quantile"):
            return False
    raise InvalidParameterError(f"无法识别的 mode：{mode!r}")


def _assemble(mapped, width, height, was_1d):
    """把扁平结果还原成嵌套列表（或一维输入对应的一维列表）。"""
    if was_1d:
        return mapped
    if width == 1:
        return [[v] for v in mapped]
    return [mapped[i : i + width] for i in range(0, height * width, width)]


# ---------------------------------------------------------------------------
# 核心实现
# ---------------------------------------------------------------------------
class Normalize:
    """分位拉伸归一化（全局 + 可选的分块 CLAHE 式局部拉伸）。

    Parameters
    ----------
    lo, hi:
        两个分位点，需满足 ``0 <= lo < hi <= 1``。
    blocks / tiles / grid:
        可选。局部模式的块网格：正整数表示每轴块数，``(ny, nx)`` 表示两轴
        各自的块数。给定时 ``apply`` 默认走局部模式。
    block_size / tile_size:
        可选。以像素为单位的块边长（正数或 ``(h, w)`` 二元组），与 ``blocks``
        互斥；给定后 ``apply`` 默认走局部模式。
    clip_limit:
        对比度限制参数，预留给 CLAHE 语义；接受并保存，不改变本实现的
        分位线性映射（保证单调性与端点性质）。
    """

    def __init__(self, lo, hi, blocks=None, clip_limit=None, **options):
        if not _is_number(lo) or not _is_number(hi):
            raise NormalizeTypeError("lo / hi 必须是 int 或 float")
        lo = float(lo)
        hi = float(hi)
        if not math.isfinite(lo) or not math.isfinite(hi):
            raise InvalidParameterError("lo / hi 必须是有限数")
        if not (0.0 <= lo <= 1.0) or not (0.0 <= hi <= 1.0):
            raise InvalidParameterError(
                f"lo / hi 必须位于 [0, 1]，实际为 lo={lo!r}, hi={hi!r}"
            )
        if not (lo < hi):
            raise InvalidParameterError(
                f"要求 lo < hi，实际为 lo={lo!r}, hi={hi!r}"
            )
        self.lo = lo
        self.hi = hi

        grid = _pop_option(options, _GRID_KEYS)
        if blocks is not None:
            grid = blocks
        size = _pop_option(options, _SIZE_KEYS)
        mode = _pop_option(options, _MODE_KEYS)
        if clip_limit is None:
            clip_limit = options.pop("clip_limit", None)
        if clip_limit is not None:
            clip_limit = _parse_positive_number(clip_limit, "clip_limit")

        if grid is not None and size is not None:
            raise InvalidParameterError("blocks 与 block_size 不能同时指定")
        if grid is not None:
            _parse_grid(grid)  # 尽早校验
            self.blocks = grid
        else:
            self.blocks = None
        if size is not None:
            _parse_size(size)  # 尽早校验
            self.block_size = size
        else:
            self.block_size = None

        self.clip_limit = clip_limit
        local = _normalize_mode(mode)
        if local is False and (grid is not None or size is not None):
            raise InvalidParameterError("mode='global' 与块参数不能同时指定")
        if local is None:
            local = grid is not None or size is not None
        self._local_mode = local

    # -- 查询 ---------------------------------------------------------------
    def quantile(self, image, q):
        """返回图像像素的 ``q`` 分位值（线性插值，float）。"""
        q = _check_quantile(q)
        _, _, flat, _ = _coerce_image(image)
        flat.sort()
        return float(_quantile_sorted(flat, q))

    def bounds(self, image):
        """返回全局 ``(lo, hi)`` 分位点对应的像素值对（float, float）。"""
        _, _, flat, _ = _coerce_image(image)
        flat.sort()
        return (
            float(_quantile_sorted(flat, self.lo)),
            float(_quantile_sorted(flat, self.hi)),
        )

    quantile_range = bounds
    range = bounds
    q = quantile
    percentile = quantile

    def quantiles(self, image, qs):
        """一次遍历返回多个分位值（``float`` 列表，与 ``qs`` 等长）。"""
        checked = [_check_quantile(q) for q in qs]
        _, _, flat, _ = _coerce_image(image)
        flat.sort()
        return [float(_quantile_sorted(flat, q)) for q in checked]

    def block_quantiles(self, image, blocks=None, **options):
        """返回分块后的每块 ``(q_lo, q_hi)`` 值（嵌套列表，下标 ``[i][j]``）。

        退化块（块内 ``q_hi <= q_lo``）返回 ``(a, a)``，对应恒等映射。
        """
        grid = _pop_option(options, _GRID_KEYS)
        size = _pop_option(options, _SIZE_KEYS)
        if blocks is not None:
            grid = blocks
        if grid is None:
            grid = self.blocks
        if size is None:
            size = self.block_size
        if grid is not None and size is not None:
            raise InvalidParameterError("blocks 与 block_size 不能同时指定")
        if grid is not None:
            _parse_grid(grid)
        if size is not None:
            _parse_size(size)
        height, width, flat, _ = _coerce_image(image)
        return self._block_bounds(flat, height, width, grid, size)

    # -- 变换 ---------------------------------------------------------------
    def apply(self, image, blocks=None, clip_limit=None, **options):
        """拉伸图像。

        未指定块参数时执行全局分位拉伸；指定后（或构造时给出块参数）
        执行分块（CLAHE 式）拉伸，块间双线性融合。
        """
        grid = _pop_option(options, _GRID_KEYS)
        size = _pop_option(options, _SIZE_KEYS)
        mode = _pop_option(options, _MODE_KEYS)
        if clip_limit is None:
            clip_limit = options.pop("clip_limit", None)

        if blocks is not None:
            grid = blocks
        if grid is None:
            grid = self.blocks
        if size is None:
            size = self.block_size
        if grid is not None and size is not None:
            raise InvalidParameterError("blocks 与 block_size 不能同时指定")
        # 无论最终走哪种模式，显式给出的块参数都必须合法。
        if grid is not None:
            _parse_grid(grid)
        if size is not None:
            _parse_size(size)

        mode_local = _normalize_mode(mode) if mode is not None else None
        if mode_local is False:
            # 显式要求全局模式时忽略块参数。
            grid = None
            size = None
            local = False
        elif mode_local is True or grid is not None or size is not None:
            # 调用时给出块参数即走局部模式。
            local = True
        else:
            local = self._local_mode
        if local:
            if grid is None and size is None:
                grid = 8

        height, width, flat, was_1d = _coerce_image(image)
        if grid is None and size is None:
            mapped = self._global_map(flat)
        else:
            if size is not None:
                ny, nx = _grid_from_size(size, height, width)
            else:
                ny, nx = _parse_grid(grid)
                ny = min(ny, height)
                nx = min(nx, width)
            mapped = self._local_map(flat, height, width, ny, nx)
        return _assemble(mapped, width, height, was_1d)

    # 常用别名：stretch/transform/normalize 与 apply 等价。
    def stretch(self, image, blocks=None, clip_limit=None, **options):
        """``apply`` 的别名（参数同 :meth:`apply`）。"""
        return self.apply(image, blocks=blocks, clip_limit=clip_limit, **options)

    transform = stretch
    normalize = stretch
    __call__ = stretch

    def apply_global(self, image, **options):
        """仅全局分位拉伸（忽略实例上的块设置）。"""
        height, width, flat, was_1d = _coerce_image(image)
        mapped = self._global_map(flat)
        return _assemble(mapped, width, height, was_1d)

    global_apply = apply_global
    global_stretch = apply_global
    stretch_global = apply_global

    def apply_local(self, image, blocks=None, clip_limit=None, **options):
        """分块（CLAHE 式）拉伸，块间双线性融合。

        块参数除位置参数 ``blocks`` 外，也可用 ``tiles`` / ``grid`` /
        ``block_size`` 等关键字（见 :meth:`apply`）。未给出任何块参数时
        使用实例上的块设置，否则默认 ``8x8``。
        """
        has_grid = blocks is not None or any(k in options for k in _GRID_KEYS)
        has_size = any(k in options for k in _SIZE_KEYS)
        if not has_grid and not has_size and self.blocks is None and self.block_size is None:
            blocks = 8
        if blocks is not None:
            options["blocks"] = blocks
        options.setdefault("mode", True)
        return self.apply(image, clip_limit=clip_limit, **options)

    local = apply_local
    local_apply = apply_local
    apply_tiled = apply_local
    apply_tiles = apply_local
    apply_clahe = apply_local
    clahe = apply_local
    local_stretch = apply_local
    stretch_local = apply_local
    tile_stretch = apply_local

    def apply_block(self, image, block=None, **options):
        """``apply_local`` 的别名，块参数支持所有别名。"""
        return self.apply(image, blocks=block, **options)

    per_block = apply_block

    # -- 内部实现 -----------------------------------------------------------
    def _quantile_pair(self, values):
        """返回一组像素值的 ``(q_lo, q_hi)``；常量输入直接给出相同值。"""
        vmin = min(values)
        if vmin == max(values):
            return vmin, vmin
        values = sorted(values)
        return (
            _quantile_sorted(values, self.lo),
            _quantile_sorted(values, self.hi),
        )

    def _global_map(self, flat):
        """全局分位拉伸后的扁平结果。"""
        a, b = self._quantile_pair(flat)
        if not (b > a):
            # 退化区间：恒等映射，常量图原样输出，不放大噪声。
            return flat
        span = b - a
        # 除法（正除数）保持 IEEE 单调性，且使端点精确映射到 0 / 1。
        return [0.0 if (t := (v - a) / span) < 0.0 else (1.0 if t > 1.0 else t)
                for v in flat]

    def _block_bounds(self, flat, height, width, grid, size):
        """计算分块后的每块 ``(a, b)`` 值。"""
        if size is not None:
            ny, nx = _grid_from_size(size, height, width)
        elif grid is not None:
            ny, nx = _parse_grid(grid)
            ny = min(ny, height)
            nx = min(nx, width)
        else:
            ny = nx = 1
        ys = _edges(height, ny)
        xs = _edges(width, nx)
        out = []
        for i in range(ny):
            r0 = ys[i]
            r1 = ys[i + 1]
            row = []
            for j in range(nx):
                c0 = xs[j]
                c1 = xs[j + 1]
                vals = []
                for r in range(r0, r1):
                    base = r * width
                    vals.extend(flat[base + c0 : base + c1])
                row.append(self._quantile_pair(vals))
            out.append(row)
        return out

    def _local_map(self, flat, height, width, ny, nx):
        """分块拉伸 + 双线性融合后的扁平结果。"""
        # 每像素一块时每块必然退化（q_lo == q_hi），整体为恒等映射。
        if ny >= height and nx >= width:
            return flat
        # 整幅常量图：所有块都退化，融合结果同样是恒等映射。
        if min(flat) == max(flat):
            return flat

        ys = _edges(height, ny)
        xs = _edges(width, nx)

        # 每块的 (a, inv)；退化块用恒等映射 (a=0, inv=1)。
        a_grid = []
        inv_grid = []
        for i in range(ny):
            r0 = ys[i] * width
            r1 = ys[i + 1] * width
            a_row = []
            inv_row = []
            for j in range(nx):
                c0 = xs[j]
                c1 = xs[j + 1]
                if c0 == 0 and c1 == width:
                    vals = flat[r0:r1]
                else:
                    vals = []
                    for r in range(ys[i], ys[i + 1]):
                        base = r * width
                        vals.extend(flat[base + c0 : base + c1])
                a, b = self._quantile_pair(vals)
                if b > a:
                    a_row.append(a)
                    inv_row.append(1.0 / (b - a))
                else:
                    a_row.append(0.0)
                    inv_row.append(1.0)
            a_grid.append(a_row)
            inv_grid.append(inv_row)

        # 每行按「参与插值的块列对 (j0, j1) 相同」切成连续小段；段内只有
        # 权重随列变化。权重只在 x 方向、与行无关，因此只构建一次；行方向的
        # 权重按区段惰性求值，避免为 100 万行分配权重表。
        xseg = []
        for start, stop, j0, j1, c0, span in _axis_segments(width, nx):
            if j0 == j1:
                weights = [0.0] * (stop - start)
            else:
                weights = [(start + k + 0.5 - c0) / span for k in range(stop - start)]
            xseg.append((start, stop, j0, j1, weights))
        yseg = _axis_segments(height, ny)

        # 单列图像（width == 1）：只有一个 x 块，融合退化为「上行块之间按行
        # 权重插值」，无需逐像素构造列表，单独走快路径。
        if width == 1:
            return self._local_map_column(flat, height, a_grid, inv_grid)

        mapped = [0.0] * (height * width)
        idx = 0
        seg_i = 0
        seg = yseg[0]
        ystop = seg[1]
        for r in range(height):
            if r >= ystop:
                seg_i += 1
                seg = yseg[seg_i]
                ystop = seg[1]
            _, _, i0, i1, c0, span = seg
            if i0 == i1:
                wy = 0.0
            else:
                wy = (r + 0.5 - c0) / span
            a0 = a_grid[i0]
            a1 = a_grid[i1]
            inv0 = inv_grid[i0]
            inv1 = inv_grid[i1]
            if wy == 0.0:
                wy0 = 1.0
            else:
                wy0 = 1.0 - wy
            row = flat[idx : idx + width]
            out_row = [0.0] * width
            for start, stop, j0, j1, weights in xseg:
                a00 = a0[j0]
                i00 = inv0[j0]
                a01 = a0[j1]
                i01 = inv0[j1]
                b00 = a1[j0]
                k00 = inv1[j0]
                b01 = a1[j1]
                k01 = inv1[j1]
                out_row[start:stop] = [
                    wy0
                    * (
                        (1.0 - t) * (0.0 if (x := (v - a00) * i00) < 0.0 else (1.0 if x > 1.0 else x))
                        + t * (0.0 if (y := (v - a01) * i01) < 0.0 else (1.0 if y > 1.0 else y))
                    )
                    + wy
                    * (
                        (1.0 - t) * (0.0 if (z := (v - b00) * k00) < 0.0 else (1.0 if z > 1.0 else z))
                        + t * (0.0 if (u := (v - b01) * k01) < 0.0 else (1.0 if u > 1.0 else u))
                    )
                    for v, t in zip(row[start:stop], weights)
                ]
            mapped[idx : idx + width] = out_row
            idx += width
        return mapped

    def _local_map_column(self, flat, height, a_grid, inv_grid):
        """单列图像的局部拉伸（x 方向只有一个块，只需按行插值）。"""
        ny = len(a_grid)
        yseg = _axis_segments(height, ny)
        mapped = [0.0] * height
        seg_i = 0
        seg = yseg[0]
        ystop = seg[1]
        for r in range(height):
            if r >= ystop:
                seg_i += 1
                seg = yseg[seg_i]
                ystop = seg[1]
            _, _, i0, i1, c0, span = seg
            v = flat[r]
            a0 = a_grid[i0][0]
            k0 = inv_grid[i0][0]
            t = (v - a0) * k0
            t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
            if i0 == i1:
                mapped[r] = t
            else:
                wy = (r + 0.5 - c0) / span
                a1 = a_grid[i1][0]
                k1 = inv_grid[i1][0]
                u = (v - a1) * k1
                u = 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)
                mapped[r] = t + (u - t) * wy
        return mapped


class LocalNormalize(Normalize):
    """默认走局部（CLAHE 式）分块拉伸的 :class:`Normalize`。"""

    def __init__(self, lo, hi, blocks=8, clip_limit=None, **options):
        super().__init__(lo, hi, blocks=blocks, clip_limit=clip_limit, **options)
        self._local_mode = True


class GlobalNormalize(Normalize):
    """仅做全局分位拉伸的 :class:`Normalize`（``apply`` 忽略块参数）。"""

    def apply(self, image, blocks=None, clip_limit=None, **options):
        return self.apply_global(image)

    stretch = apply
    transform = apply
    normalize = apply
    __call__ = apply


# 语义别名：直接按用途命名的入口。
CLAHE = LocalNormalize
Local = LocalNormalize
Global = GlobalNormalize


# ---------------------------------------------------------------------------
# 模块级便捷函数
# ---------------------------------------------------------------------------
def normalize(image, lo=0.05, hi=0.95, blocks=None, **options):
    """全局分位拉伸（给出 ``blocks`` / ``block_size`` 等时走局部模式）。"""
    return Normalize(lo, hi, blocks=blocks, **options).apply(image)


def stretch(image, lo=0.05, hi=0.95, blocks=None, **options):
    """``normalize`` 的别名。"""
    return Normalize(lo, hi, blocks=blocks, **options).apply(image)


# 语义别名：不同命名习惯的调用方均可直接使用。
global_stretch = normalize
stretch_quantile = normalize
Normalizer = Normalize
QuantileStretch = Normalize


def local_stretch(image, lo=0.05, hi=0.95, blocks=8, clip_limit=None, **options):
    """``apply_local`` 的别名。"""
    return apply_local(image, lo, hi, blocks, clip_limit, **options)


local_normalize = local_stretch


def quantile(image, q):
    """图像像素的 ``q`` 分位值。"""
    return Normalize(0.0, 1.0).quantile(image, q)


def quantiles(image, qs):
    """图像像素的多个分位值（一次遍历）。"""
    return Normalize(0.0, 1.0).quantiles(image, qs)


# 常见命名：``percentile`` 与 ``quantile`` 等价（参数均为 [0, 1] 上的比例）。
percentile = quantile


def apply_local(image, lo=0.05, hi=0.95, blocks=8, clip_limit=None, **options):
    """分块（CLAHE 式）归一化。

    块参数也可用 ``tiles`` / ``grid`` / ``block_size`` 等关键字给出；
    显式给出 ``block_size`` 时优先于 ``blocks``。
    """
    has_grid = any(k in options for k in _GRID_KEYS)
    has_size = any(k in options for k in _SIZE_KEYS)
    if has_size:
        blocks = None
    elif not has_grid and blocks is not None:
        options["blocks"] = blocks
    options.setdefault("mode", True)
    return Normalize(lo, hi, clip_limit=clip_limit, **options).apply(image)


def clahe(image, lo=0.05, hi=0.95, blocks=8, clip_limit=None, **options):
    """``apply_local`` 的别名。"""
    return apply_local(image, lo, hi, blocks, clip_limit, **options)
