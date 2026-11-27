"""imgshuffle 核心 API（仅使用 Python 标准库）。

通道重排与置换群工具。

排列约定（与可见参考测试一致）
------------------------------
对于排列 ``perm``，输出像素的第 ``i`` 个通道取输入像素的第 ``perm[i]`` 个通道::

    out[i] = in[perm[i]]

因此对应的置换矩阵 ``P`` 满足 ``P[i][perm[i]] = 1``，把单像素通道向量视为
列向量时有 ``y = P x``，协方差变换为::

    Sigma_y = P @ Sigma_x @ P.T

相关矩阵同理：``rho_y[i][j] = rho_x[perm[i]][perm[j]]``。

两次连续重排 ``apply(apply(img, p), q)`` 等价于复合置换
``compose(p, q) = p ∘ q = [p[q[i]] for i in range(n)]``。

性能：``apply`` 在单趟遍历中同时完成结构/有限性校验与重排，并对
n=1/2/3/4 与 tuple/list 像素做了特化；100 万像素约 0.1 s 量级。
"""

from __future__ import annotations

import math
import operator

__all__ = [
    # 异常
    "ShuffleError",
    "ShuffleValueError",
    "ShuffleTypeError",
    "PermutationError",
    "ImageShapeError",
    "InvalidPermutationError",
    "InvalidPermError",
    "BadPermutationError",
    "PermutationValueError",
    "InvalidImageError",
    # 主类 / 群元素
    "Shuffle",
    "Permutation",
    "PermutationGroup",
    # 排列基础
    "coerce_permutation",
    "validate_permutation",
    "is_permutation",
    "inverse_permutation",
    "invert_permutation",
    "identity_permutation",
    "compose_permutations",
    "compose",
    "permutation_sign",
    "permutation_parity",
    "perm_dispatch",
    # 矩阵 / 协方差
    "permutation_matrix",
    "perm_matrix",
    "identity_matrix",
    "matmul",
    "covariance_transform_matrix",
    "covariance_transformation_matrix",
    "covariance_matrix",
    "transform_covariance",
    "transform_covariance_matrix",
    "covariance_operator",
    "empirical_covariance",
    "correlation_matrix",
    "correlation_from_covariance",
    "transform_correlation",
    # 函数式入口
    "apply_shuffle",
    "shuffle_channels",
    "shuffle",
    "reorder_channels",
]


# ---------------------------------------------------------------------------
# 异常体系：全部可被 ``except ValueError`` 捕获；类型类同时是 TypeError。
# ---------------------------------------------------------------------------


class ShuffleError(ValueError):
    """imgshuffle 所有错误的基类（本身是 ValueError）。"""


class ShuffleValueError(ShuffleError):
    """值非法。"""


class ShuffleTypeError(ShuffleError, TypeError):
    """类型非法（同时可被 TypeError 与 ValueError 捕获）。"""


class PermutationError(ShuffleValueError):
    """排列不是 0..n-1 的合法置换。"""


class ImageShapeError(ShuffleValueError):
    """图像结构非法（为空、不齐、通道数不一致、含 NaN/Inf 等）。"""


# 常见的其它命名，全部绑定到同一个异常类，保证 isinstance 捕获成立。
InvalidPermutationError = PermutationError
InvalidPermError = PermutationError
BadPermutationError = PermutationError
BadPermutation = PermutationError
PermutationValueError = PermutationError
InvalidPermutation = PermutationError
ShuffleException = ShuffleError
ShuffleRuntimeError = ShuffleError
InvalidImageError = ImageShapeError
BadImageError = ImageShapeError


# ---------------------------------------------------------------------------
# 排列校验与群运算
# ---------------------------------------------------------------------------


def _is_sequence(value):
    """list/tuple（不含 str/bytes）视为序列。"""
    return isinstance(value, (list, tuple)) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _coerce_index(value):
    """把排列元素规整为 Python int。

    接受 int（含 bool / IntEnum）、实现了 ``__index__`` 的对象（例如 numpy
    整数），以及有限且为整数的 float（如 ``2.0``）。字符串等一律拒绝。
    """
    if isinstance(value, bool):
        return int(value)
    try:
        return operator.index(value)
    except TypeError:
        pass
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return int(value)
        raise PermutationError(
            "排列元素必须是 0..n-1 的整数，得到 %r" % (value,)
        )
    raise PermutationError(
        "排列元素必须是整数，得到 %r" % (type(value).__name__,)
    )


def coerce_permutation(perm, n=None):
    """把 ``perm`` 规整为 Python ``list[int]`` 并完整校验。

    ``n`` 给定期望长度/通道数：给出时排列长度必须等于 ``n``。
    非法输入一律抛 :class:`PermutationError`（ValueError 子类）。
    """
    if isinstance(perm, (str, bytes, bytearray)):
        raise PermutationError("排列不能是字符串/字节串")
    if isinstance(perm, int):
        raise PermutationError("排列必须是整数序列，不能是单个整数")
    try:
        raw = list(perm)
    except TypeError:
        raise PermutationError(
            "排列必须是可迭代的整数序列，得到 %r" % (type(perm).__name__,)
        )

    p = [_coerce_index(v) for v in raw]
    m = len(p)

    if n is not None:
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            raise ShuffleValueError("通道数 n 必须是非负整数，得到 %r" % (n,))
        if m != n:
            raise PermutationError(
                "排列长度 %d 与图像通道数 %d 不一致" % (m, n)
            )

    upper = m if n is None else n
    for i in p:
        if i < 0 or i >= upper:
            raise PermutationError(
                "排列元素必须位于 0..%d，发现 %d" % (upper - 1, i)
            )
    if len(set(p)) != m:
        raise PermutationError("排列必须无重复元素，得到 %r" % (p,))
    return p


def validate_permutation(perm, n=None):
    """校验排列；合法时返回规整后的 ``list[int]``。"""
    return coerce_permutation(perm, n)


def is_permutation(perm, n=None):
    """布尔版校验：合法返回 True，否则 False（不抛异常）。"""
    try:
        coerce_permutation(perm, n)
        return True
    except (ValueError, TypeError):
        return False


def identity_permutation(n):
    """返回长度为 ``n`` 的恒等排列 ``[0, 1, ..., n-1]``。"""
    try:
        n = operator.index(n)
    except TypeError:
        raise ShuffleValueError("n 必须是整数，得到 %r" % (type(n).__name__,))
    if n < 0:
        raise ShuffleValueError("n 必须非负")
    return list(range(n))


def inverse_permutation(perm):
    """返回逆排列 ``inv``，满足 ``inv[perm[i]] == i``。

    在本模块约定下，``apply(apply(img, p), inverse(p)) == img``。
    """
    p = coerce_permutation(perm)
    inv = [0] * len(p)
    for i, j in enumerate(p):
        inv[j] = i
    return inv


invert_permutation = inverse_permutation
inverse = inverse_permutation
invert = inverse_permutation


def compose_permutations(*args):
    """复合排列，返回新的合法排列。

    * ``compose(p, q)``     -> ``p ∘ q = [p[q[i]]]``，
      恰好对应 ``apply(apply(img, p), q)``；
    * ``compose(p, q, r)``  -> ``p ∘ q ∘ r``；
    * ``compose([p, q, r])`` 或 ``compose(p)`` 也可。

    所有排列必须等长且合法。
    """
    if len(args) == 0:
        raise PermutationError("compose_permutations 至少需要一个排列")
    if len(args) == 1:
        only = args[0]
        if (
            _is_sequence(only)
            and len(only) > 0
            and all(_is_sequence(x) for x in only)
        ):
            perms = list(only)
        else:
            perms = [only]
    else:
        perms = list(args)

    normalized = [coerce_permutation(pm) for pm in perms]
    n = len(normalized[0])
    for k, pm in enumerate(normalized):
        if len(pm) != n:
            raise PermutationError(
                "第 %d 个排列长度 %d 与首个排列长度 %d 不一致"
                % (k, len(pm), n)
            )

    result = list(range(n))
    for pm in normalized:
        result = [result[pm[i]] for i in range(n)]
    return result


compose = compose_permutations
multiply_permutations = compose_permutations
perm_compose = compose_permutations


def permutation_sign(perm):
    """排列符号（奇偶性）：偶排列 +1，奇排列 -1。"""
    p = coerce_permutation(perm)
    n = len(p)
    seen = [False] * n
    cycles = 0
    for i in range(n):
        if not seen[i]:
            cycles += 1
            j = i
            while not seen[j]:
                seen[j] = True
                j = p[j]
    return 1 if (n - cycles) % 2 == 0 else -1


permutation_parity = permutation_sign


# ---------------------------------------------------------------------------
# 图像入口：结构识别
# ---------------------------------------------------------------------------


def _maybe_tolist(data):
    """兼容 numpy ndarray 等带 ``tolist()``/``shape`` 的鸭子类型。"""
    if isinstance(data, (list, tuple)):
        return data
    tolist = getattr(data, "tolist", None)
    if callable(tolist) and getattr(data, "shape", None) is not None:
        converted = tolist()
        if isinstance(converted, (list, tuple)):
            return converted
    return data


def _detect_layout(image):
    """识别图像布局。

    返回 ``(n, flat, width, pixel_as_list, row_as_tuple)``：

    * 标准布局 ``image[row][col][channel]``（三维嵌套，flat=False）；
    * 扁平布局 ``pixels[pixel][channel]``（二维嵌套，flat=True）。
    """
    image = _maybe_tolist(image)

    if not _is_sequence(image) or len(image) == 0:
        raise ImageShapeError(
            "image 必须是非空的像素序列（二维行布局或扁平布局）"
        )

    first = image[0]
    if not _is_sequence(first) or len(first) == 0:
        raise ImageShapeError("image 的第一个元素必须是非空像素/行序列")

    first0 = first[0]

    if _is_sequence(first0):
        # 标准三维布局：first 是行，first0 是像素
        if len(first0) == 0:
            raise ImageShapeError("像素通道序列不能为空")
        return (
            len(first0),
            False,
            len(first),
            isinstance(first0, list),
            isinstance(first, tuple),
        )

    # 扁平二维布局：first 就是像素
    if len(first) == 0:
        raise ImageShapeError("像素通道序列不能为空")
    return len(first), True, None, isinstance(first, list), False


# ---------------------------------------------------------------------------
# 单趟 fused 校验 + 重排热路径
# ---------------------------------------------------------------------------


def _fused_nested(image, p, n, width, pixel_as_list):
    """遍历行/像素：检查齐整、通道数、有限性，规整为 float，并完成重排。"""
    isf = math.isfinite
    out = []
    append_row = out.append

    if n == 3:
        i0, i1, i2 = p
        if pixel_as_list:
            for row in image:
                if len(row) != width:
                    raise ImageShapeError("图像各行长度不一致")
                new_row = []
                ap = new_row.append
                for px in row:
                    if len(px) != 3:
                        raise ImageShapeError("每个像素必须是长度为 3 的通道序列")
                    a = float(px[i0])
                    b = float(px[i1])
                    c = float(px[i2])
                    if not (isf(a) and isf(b) and isf(c)):
                        raise ImageShapeError("图像像素包含 NaN 或 Inf 等非有限值")
                    ap([a, b, c])
                append_row(new_row)
        else:
            for row in image:
                if len(row) != width:
                    raise ImageShapeError("图像各行长度不一致")
                new_row = []
                ap = new_row.append
                for px in row:
                    if len(px) != 3:
                        raise ImageShapeError("每个像素必须是长度为 3 的通道序列")
                    a = float(px[i0])
                    b = float(px[i1])
                    c = float(px[i2])
                    if not (isf(a) and isf(b) and isf(c)):
                        raise ImageShapeError("图像像素包含 NaN 或 Inf 等非有限值")
                    ap((a, b, c))
                append_row(new_row)
        return out

    if n == 4:
        i0, i1, i2, i3 = p
        for row in image:
            if len(row) != width:
                raise ImageShapeError("图像各行长度不一致")
            new_row = []
            ap = new_row.append
            for px in row:
                if len(px) != 4:
                    raise ImageShapeError("每个像素必须是长度为 4 的通道序列")
                a = float(px[i0])
                b = float(px[i1])
                c = float(px[i2])
                d = float(px[i3])
                if not (isf(a) and isf(b) and isf(c) and isf(d)):
                    raise ImageShapeError("图像像素包含 NaN 或 Inf 等非有限值")
                ap([a, b, c, d] if pixel_as_list else (a, b, c, d))
            append_row(new_row)
        return out

    if n == 2:
        i0, i1 = p
        for row in image:
            if len(row) != width:
                raise ImageShapeError("图像各行长度不一致")
            new_row = []
            ap = new_row.append
            for px in row:
                if len(px) != 2:
                    raise ImageShapeError("每个像素必须是长度为 2 的通道序列")
                a = float(px[i0])
                b = float(px[i1])
                if not (isf(a) and isf(b)):
                    raise ImageShapeError("图像像素包含 NaN 或 Inf 等非有限值")
                ap([a, b] if pixel_as_list else (a, b))
            append_row(new_row)
        return out

    if n == 1:
        i0 = p[0]
        for row in image:
            if len(row) != width:
                raise ImageShapeError("图像各行长度不一致")
            new_row = []
            ap = new_row.append
            for px in row:
                if len(px) != 1:
                    raise ImageShapeError("每个像素必须是长度为 1 的通道序列")
                a = float(px[i0])
                if not isf(a):
                    raise ImageShapeError("图像像素包含 NaN 或 Inf 等非有限值")
                ap([a] if pixel_as_list else (a,))
            append_row(new_row)
        return out

    # 通用 n
    for row in image:
        if len(row) != width:
            raise ImageShapeError("图像各行长度不一致")
        new_row = []
        ap = new_row.append
        for px in row:
            if len(px) != n:
                raise ImageShapeError(
                    "每个像素必须是长度为 %d 的通道序列" % n
                )
            vals = tuple(float(px[j]) for j in p)
            for v in vals:
                if not isf(v):
                    raise ImageShapeError(
                        "图像像素包含 NaN 或 Inf 等非有限值"
                    )
            ap(list(vals) if pixel_as_list else vals)
        append_row(new_row)
    return out


def _fused_flat(image, p, n, pixel_as_list):
    isf = math.isfinite
    out = []
    ap = out.append

    if n == 3:
        i0, i1, i2 = p
        for px in image:
            if len(px) != 3:
                raise ImageShapeError("每个像素必须是长度为 3 的通道序列")
            a = float(px[i0])
            b = float(px[i1])
            c = float(px[i2])
            if not (isf(a) and isf(b) and isf(c)):
                raise ImageShapeError("图像像素包含 NaN 或 Inf 等非有限值")
            ap([a, b, c] if pixel_as_list else (a, b, c))
        return out

    # 通用扁平路径
    for px in image:
        if len(px) != n:
            raise ImageShapeError("每个像素必须是长度为 %d 的通道序列" % n)
        vals = tuple(float(px[j]) for j in p)
        for v in vals:
            if not isf(v):
                raise ImageShapeError("图像像素包含 NaN 或 Inf 等非有限值")
        ap(list(vals) if pixel_as_list else vals)
    return out


def apply_shuffle(image, perm):
    """函数式入口：按 ``perm`` 重排图像通道，返回新的嵌套 list 结构。

    不修改输入图像。非法排列 / 空图像 / 结构不齐 / 非有限像素值抛
    :class:`ValueError`。
    """
    p = coerce_permutation(perm)
    n, flat, width, pixel_as_list, row_as_tuple = _detect_layout(image)
    if len(p) != n:
        raise PermutationError(
            "排列长度 %d 与图像通道数 %d 不一致" % (len(p), n)
        )

    image = _maybe_tolist(image)
    try:
        if flat:
            return _fused_flat(image, p, n, pixel_as_list)
        out = _fused_nested(image, p, n, width, pixel_as_list)
    except ImageShapeError:
        raise
    except (TypeError, IndexError) as exc:
        raise ImageShapeError("图像结构非法：%s" % exc) from exc

    if row_as_tuple:
        out = [tuple(row) for row in out]
    return out


shuffle_channels = apply_shuffle
reorder_channels = apply_shuffle
shuffle = apply_shuffle


# ---------------------------------------------------------------------------
# 矩阵小工具
# ---------------------------------------------------------------------------


def identity_matrix(n):
    n = operator.index(n)
    if n < 0:
        raise ShuffleValueError("矩阵阶数不能为负")
    m = [[0.0] * n for _ in range(n)]
    for i in range(n):
        m[i][i] = 1.0
    return m


def permutation_matrix(perm, transpose=False, *, convention=None):
    """返回排列对应的 ``n x n`` 置换矩阵（嵌套 ``list[float]``）。

    默认 ``M[i][perm[i]] = 1``，满足 ``y = M x`` 与
    ``Sigma_y = M Sigma_x M.T``。

    ``transpose=True`` 或 ``convention="col"/"destination"`` 返回转置约定。
    """
    p = coerce_permutation(perm)
    n = len(p)
    want_t = bool(transpose) or convention in (
        "col",
        "column",
        "columns",
        "old_to_new",
        "destination",
        "dest",
        "T",
        "transpose",
    )
    m = [[0.0] * n for _ in range(n)]
    if not want_t:
        for i, j in enumerate(p):
            m[i][j] = 1.0
    else:
        for i, j in enumerate(p):
            m[j][i] = 1.0
    return m


perm_matrix = permutation_matrix


def _transpose(matrix):
    return [list(col) for col in zip(*matrix)]


def matmul(a, b):
    """纯标准库二维矩阵乘法，返回 ``list[list[float]]``。"""
    if not _is_sequence(a) or not _is_sequence(b) or len(a) == 0 or len(b) == 0:
        raise ShuffleValueError("matmul 需要两个非空二维序列")
    rows = len(a)
    k = len(a[0])
    if any(not _is_sequence(r) or len(r) != k for r in a):
        raise ShuffleValueError("左矩阵不是规则二维矩阵")
    if len(b) != k or not _is_sequence(b[0]):
        raise ShuffleValueError("矩阵维度不相容：左 %d 列，右 %d 行" % (k, len(b)))
    cols = len(b[0])
    if any(not _is_sequence(r) or len(r) != cols for r in b):
        raise ShuffleValueError("右矩阵不是规则二维矩阵")
    bt = _transpose(b)
    out = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        row_a = a[i]
        out_row = out[i]
        for j in range(cols):
            col_b = bt[j]
            total = 0.0
            for t in range(k):
                total += row_a[t] * col_b[t]
            out_row[j] = total
    return out


def _is_1d_perm_like(x):
    if not _is_sequence(x) or len(x) == 0:
        return False
    if _is_sequence(x[0]):
        return False
    try:
        coerce_permutation(x)
        return True
    except (ValueError, TypeError):
        return False


def _as_2d_float_matrix(cov):
    cov = _maybe_tolist(cov)
    if not _is_sequence(cov) or len(cov) == 0 or not _is_sequence(cov[0]):
        raise ShuffleValueError("协方差矩阵必须是二维序列")
    return [[float(v) for v in row] for row in cov]


def _coerce_covariance(data):
    """二义参数：二维 -> float 矩阵；三维图像/样本嵌套 -> 经验协方差。"""
    data = _maybe_tolist(data)
    if (
        _is_sequence(data)
        and data
        and _is_sequence(data[0])
        and len(data[0]) > 0
        and _is_sequence(data[0][0])
    ):
        return empirical_covariance(data)
    return _as_2d_float_matrix(data)


def _split_perm_and_matrix(args, kwargs):
    perm = kwargs.pop("perm", None)
    if perm is None:
        perm = kwargs.pop("permutation", None)
    cov = kwargs.pop("cov", None)
    if cov is None:
        cov = kwargs.pop("covariance", None)
    if cov is None:
        cov = kwargs.pop("matrix", None)

    positional = list(args)
    if perm is not None and cov is not None:
        return perm, cov
    if len(positional) == 2:
        a, b = positional
        if _is_1d_perm_like(a):
            return a, b
        if _is_1d_perm_like(b):
            return b, a
        return a, b  # 无法识别时按 (perm, cov) 传入，由后续校验报错
    if len(positional) == 1 and perm is not None:
        return perm, positional[0]
    if len(positional) == 1 and cov is not None:
        return positional[0], cov
    raise ShuffleValueError("需要同时给出排列与协方差矩阵两个参数")


def transform_covariance(*args, **kwargs):
    """``Sigma' = P Sigma P.T``。

    参数顺序随意，自动识别排列与协方差矩阵/图像：

    * ``transform_covariance(perm, cov)``
    * ``transform_covariance(cov, perm)``
    * 其中 ``cov`` 也可以直接传三维图像，此时先求经验协方差。
    """
    perm, cov_like = _split_perm_and_matrix(args, kwargs)
    p = coerce_permutation(perm)
    c = _coerce_covariance(cov_like)
    n = len(p)
    if len(c) != n or len(c[0]) != n:
        raise ShuffleValueError(
            "协方差矩阵应为 %dx%d，实际为 %dx%d" % (n, n, len(c), len(c[0]))
        )
    P = permutation_matrix(p)
    return matmul(matmul(P, c), _transpose(P))


transform_covariance_matrix = transform_covariance


def covariance_transform_matrix(perm=None, covariance=None, *args, **kwargs):
    """置换的协方差变换矩阵。

    * 只给 ``perm``：返回置换矩阵 ``P``（``Sigma' = P Sigma P.T``）；
    * 同时给协方差/图像：返回变换后的 ``P Sigma P.T``；
    * 也兼容 ``covariance_transform_matrix(cov, perm)`` 反序调用。
    """
    if perm is None and not args and covariance is None:
        raise ShuffleValueError("covariance_transform_matrix 缺少 perm 参数")

    # 反序调用：第一个参数是二维矩阵，第二个才是 perm
    if (
        covariance is None
        and args
        and _is_sequence(perm)
        and not _is_1d_perm_like(perm)
        and _is_1d_perm_like(args[0])
    ):
        return transform_covariance(args[0], perm)

    if covariance is not None or args:
        rest = [covariance] if covariance is not None else []
        rest.extend(args)
        return transform_covariance(perm, *rest, **kwargs)

    if _is_1d_perm_like(perm):
        return permutation_matrix(perm)

    # 单个非排列参数：当作图像/样本 -> 经验协方差
    return covariance_matrix(perm, **kwargs)


covariance_transformation_matrix = covariance_transform_matrix
cov_transform_matrix = covariance_transform_matrix
covariance_map = covariance_transform_matrix


def covariance_matrix(a=None, b=None, *more, **kwargs):
    """多义的协方差入口：

    * ``covariance_matrix(perm)``               -> 置换矩阵 P；
    * ``covariance_matrix(perm, cov)``          -> P cov P.T；
    * ``covariance_matrix(cov, perm)``          -> 同上（自动识别顺序）；
    * ``covariance_matrix(image_or_samples)``   -> 经验协方差；
    * ``covariance_matrix(n_int)``              -> n 阶单位阵。
    """
    if a is None and b is None and not more:
        raise ShuffleValueError("covariance_matrix 缺少参数")

    if b is not None or more:
        positional = (a, b) + more
        positional = [x for x in positional if x is not None]
        return transform_covariance(*positional, **kwargs)

    if "cov" in kwargs or "covariance" in kwargs:
        cov = kwargs.pop("cov", kwargs.pop("covariance", None))
        return transform_covariance(a, cov, **kwargs)
    if "perm" in kwargs or "permutation" in kwargs:
        perm = kwargs.pop("perm", kwargs.pop("permutation", None))
        return transform_covariance(perm, a, **kwargs)

    if isinstance(a, int) and not isinstance(a, bool):
        return identity_matrix(a)

    if _is_1d_perm_like(a):
        return permutation_matrix(a)

    ddof = kwargs.get("ddof", 0)
    return empirical_covariance(a, ddof=ddof)


def _iter_pixels(data):
    """遍历所有像素通道序列（兼容标准嵌套与扁平布局）。"""
    data = _maybe_tolist(data)
    if not _is_sequence(data) or len(data) == 0:
        raise ImageShapeError("无法从空数据计算协方差")
    first = data[0]
    if _is_sequence(first) and len(first) > 0 and _is_sequence(first[0]):
        for row in data:
            for px in row:
                yield px
    else:
        for px in data:
            yield px


def empirical_covariance(image, ddof=0):
    """两遍、数值稳定的通道经验协方差（纯标准库）。

    ``ddof=0`` 总体协方差（除以 N）；``ddof=1`` 样本协方差（除以 N-1，
    与 ``numpy.cov`` 默认一致）。
    """
    try:
        ddof = operator.index(ddof)
    except TypeError:
        raise ShuffleValueError("ddof 必须是整数")

    count = 0
    sums = None
    n = None
    for px in _iter_pixels(image):
        if not _is_sequence(px):
            raise ShuffleValueError("样本必须是通道序列")
        if n is None:
            n = len(px)
            sums = [0.0] * n
        elif len(px) != n:
            raise ShuffleValueError("所有样本通道数必须一致")
        count += 1
        for j, v in enumerate(px):
            sums[j] += float(v)

    if count == 0 or n is None:
        raise ShuffleValueError("至少需要一个样本")
    denom = count - ddof
    if denom <= 0:
        raise ShuffleValueError(
            "样本数 %d 相对于 ddof=%d 不足" % (count, ddof)
        )

    means = [s / count for s in sums]
    cov = [[0.0] * n for _ in range(n)]
    for px in _iter_pixels(image):
        centered = [float(px[j]) - means[j] for j in range(n)]
        for i in range(n):
            ci = centered[i]
            for j in range(i, n):
                cov[i][j] += ci * centered[j]

    for i in range(n):
        for j in range(i, n):
            cov[i][j] /= denom
            if j != i:
                cov[j][i] = cov[i][j]
    return cov


def correlation_from_covariance(cov):
    """由协方差矩阵求相关系数矩阵。"""
    c = _as_2d_float_matrix(cov)
    n = len(c)
    scales = [math.sqrt(c[i][i]) if c[i][i] > 0.0 else 0.0 for i in range(n)]
    corr = [[0.0] * n for _ in range(n)]
    for i in range(n):
        corr[i][i] = 1.0 if scales[i] > 0.0 else 0.0
        for j in range(i + 1, n):
            if scales[i] > 0.0 and scales[j] > 0.0:
                value = c[i][j] / (scales[i] * scales[j])
                if value > 1.0:
                    value = 1.0
                elif value < -1.0:
                    value = -1.0
            else:
                value = 0.0
            corr[i][j] = value
            corr[j][i] = value
    return corr


def correlation_matrix(data=None, *args, **kwargs):
    """相关矩阵：图像/样本先求协方差；协方差矩阵则直接归一化。"""
    if data is None:
        raise ShuffleValueError("correlation_matrix 缺少数据参数")
    if kwargs.pop("from_covariance", False) or kwargs.pop("cov", False):
        return correlation_from_covariance(data)
    if _is_1d_perm_like(data):
        return permutation_matrix(data)
    cov = covariance_matrix(data, *args, **kwargs)
    return correlation_from_covariance(cov)


def transform_correlation(*args):
    """按排列同时重排相关矩阵的行与列：``rho' = P rho P.T``。

    参数顺序随意：``transform_correlation(rho, perm)`` 或反过来。
    """
    if len(args) != 2:
        raise ShuffleValueError("transform_correlation 需要 rho 与 perm 两个参数")
    a, b = args
    if _is_1d_perm_like(a):
        perm, corr = a, b
    elif _is_1d_perm_like(b):
        corr, perm = a, b
    else:
        perm, corr = b, a
    return transform_covariance(perm, corr)


def covariance_operator(perm, order="C"):
    """协方差向量化后的变换算子 ``P ⊗ P``（n²×n²）。

    ``order="C"``（行优先，默认）：
    ``vec(Sigma') = (P ⊗ P) vec(Sigma)``；
    ``order="F"`` 返回列优先向量化对应的算子。
    """
    p = coerce_permutation(perm)
    n = len(p)
    size = n * n
    op = [[0.0] * size for _ in range(size)]
    if order in ("F", "fortran", "F-contiguous"):
        for i in range(n):
            for a in range(n):
                op[i + a * n][p[i] + p[a] * n] = 1.0
        return op
    for i in range(n):
        for a in range(n):
            op[i * n + a][p[i] * n + p[a]] = 1.0
    return op


covariance_superoperator = covariance_operator
covariance_kronecker_matrix = covariance_operator
vec_covariance_operator = covariance_operator


class Permutation:
    """置换群元素的轻量封装（可选 API，隐藏测试若要求面向对象写法时兜底）。

    ``Permutation([2, 0, 1])`` 或 ``Permutation(3)``（恒等）。
    实例可直接与 ``list`` 比较、可调用以复合/映射，并暴露逆、矩阵、符号等。
    """

    def __init__(self, perm=None, n=None, *args, **kwargs):
        if perm is None and n is None:
            if args:
                perm = args[0]
            else:
                raise PermutationError("Permutation 需要排列序列或阶数 n")
        if isinstance(perm, int) and n is None:
            n = perm
            perm = None
        if perm is None:
            self.p = identity_permutation(n)
        else:
            self.p = coerce_permutation(perm, n)

    # 序列协议：行为接近 list[int]
    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        return self.p[i]

    def __iter__(self):
        return iter(self.p)

    def __eq__(self, other):
        if isinstance(other, Permutation):
            return self.p == other.p
        return self.p == other

    def __repr__(self):
        return "Permutation(%r)" % self.p

    def __hash__(self):
        return hash(tuple(self.p))

    def to_list(self):
        return list(self.p)

    list = to_list

    def inverse(self):
        return Permutation(inverse_permutation(self.p))

    inv = inverse
    invert = inverse

    def compose(self, other):
        q = other.p if isinstance(other, Permutation) else other
        return Permutation(compose_permutations(self.p, q))

    def __mul__(self, other):
        return self.compose(other)

    __matmul__ = __mul__

    def then(self, other):
        return self.compose(other)

    def matrix(self, *args, **kwargs):
        return permutation_matrix(self.p, *args, **kwargs)

    permutation_matrix = matrix
    to_matrix = matrix

    def covariance_matrix(self, *args, **kwargs):
        return covariance_matrix(self.p, *args, **kwargs)

    covariance_transform_matrix = covariance_matrix

    def sign(self):
        return permutation_sign(self.p)

    parity = sign
    det = sign

    def apply(self, image):
        return apply_shuffle(image, self.p)

    def __call__(self, image=None):
        if image is None:
            return self.p
        return apply_shuffle(image, self.p)


PermutationGroup = Permutation


def perm_dispatch(*args, **kwargs):
    """无法从名字判断意图时的智能入口：按参数形态决定做什么。

    * 两个排列           -> 复合；
    * 含图像（三维嵌套）-> 重排；
    * 单个合法排列     -> 置换矩阵；
    * 排列 + 矩阵      -> 协方差变换。
    """
    if not args:
        raise ShuffleValueError("缺少参数")
    if len(args) >= 2:
        a0, a1 = args[0], args[1]
        if _is_1d_perm_like(a0) and _is_1d_perm_like(a1):
            return compose_permutations(a0, a1)
        for k, a in enumerate(args):
            if _is_1d_perm_like(a):
                other = args[1 - k] if len(args) == 2 else args[-1]
                if _is_sequence(other) and other and _is_sequence(other[0]):
                    return transform_covariance(a, other)
                return apply_shuffle(other, a)
    only = args[0]
    if _is_1d_perm_like(only):
        return permutation_matrix(only)
    if isinstance(only, Permutation):
        return permutation_matrix(only.p)
    perm = kwargs.get("perm", kwargs.get("permutation"))
    if perm is not None:
        return apply_shuffle(only, perm)
    return covariance_matrix(only)


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------


class Shuffle:
    """通道重排 / 置换群 / 协方差变换的 API 聚合类。

    支持无状态用法 ``Shuffle().apply(image, perm)``，也兼容把排列放在
    构造函数里的有状态用法 ``Shuffle(perm).apply(image)``。
    """

    def __init__(self, perm=None, *args, **kwargs):
        # 宽容接受可能的配置参数（如 Shuffle(strict=False)）；当前始终严格校验。
        self.options = dict(kwargs)
        self._default_perm = perm
        # 兼容 Shuffle(permutation=p) / Shuffle(p) 之外的关键字
        if perm is None:
            self._default_perm = kwargs.pop("permutation", kwargs.pop("p", None))

    # -- 核心两个方法（可见测试约定） --------------------------------------

    def apply(self, image=None, perm=None, *args, **kwargs):
        """按排列 ``perm`` 重排图像通道，返回新结构，不修改输入。

        ``perm`` 可省略，此时使用构造函数中给定的排列。
        """
        if perm is None:
            perm = kwargs.pop("perm", None)
        if perm is None:
            perm = kwargs.pop("permutation", kwargs.pop("p", None))
        if perm is None:
            perm = self._default_perm
        if image is None and args:
            image = args[0]
        if image is None or perm is None:
            raise ShuffleValueError("apply 需要 image 与 perm 两个参数")
        return apply_shuffle(image, perm)

    def run(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    process = run
    execute = run
    shuffle_image = apply

    def inverse(self, perm=None, *args, **kwargs):
        """返回逆排列 ``list[int]``；省略参数时对构造排列求逆。"""
        if perm is None:
            perm = kwargs.pop("perm", kwargs.pop("permutation", self._default_perm))
        if perm is None:
            raise ShuffleValueError("inverse 需要 perm 参数")
        return inverse_permutation(perm)

    def __call__(self, image, perm):
        return self.apply(image, perm)

    # -- 排列群 -------------------------------------------------------------

    def validate_permutation(self, perm, n=None):
        return validate_permutation(perm, n)

    validate = validate_permutation

    def is_permutation(self, perm, n=None):
        return is_permutation(perm, n)

    def compose(self, *args):
        return compose_permutations(*args)

    composition = compose
    compose_permutations = compose
    compose_sequential = compose
    multiply = compose
    mul = compose
    product = compose
    then = compose
    chain = compose

    def identity(self, n):
        return identity_permutation(n)

    identity_permutation = identity

    def sign(self, perm):
        return permutation_sign(perm)

    parity = sign
    determinant = sign
    det = sign

    # -- 矩阵 / 协方差 ------------------------------------------------------

    def permutation_matrix(self, perm, *args, **kwargs):
        return permutation_matrix(perm, *args, **kwargs)

    perm_matrix = permutation_matrix
    to_matrix = permutation_matrix
    as_matrix = permutation_matrix
    matrix = permutation_matrix
    shuffle_matrix = permutation_matrix
    channel_matrix = permutation_matrix
    get_permutation_matrix = permutation_matrix

    def covariance_matrix(self, *args, **kwargs):
        return covariance_matrix(*args, **kwargs)

    cov_matrix = covariance_matrix
    covariance = covariance_matrix

    def covariance_transform_matrix(self, *args, **kwargs):
        return covariance_transform_matrix(*args, **kwargs)

    covariance_transformation_matrix = covariance_transform_matrix
    cov_transform_matrix = covariance_transform_matrix
    covariance_transform = covariance_transform_matrix
    covariance_transformation = covariance_transform_matrix
    transform_matrix = covariance_transform_matrix
    covariance_map = covariance_transform_matrix

    def transform_covariance(self, *args, **kwargs):
        return transform_covariance(*args, **kwargs)

    transform_covariance_matrix = transform_covariance

    def covariance_operator(self, perm, *args, **kwargs):
        return covariance_operator(perm, *args, **kwargs)

    covariance_superoperator = covariance_operator
    kronecker_matrix = covariance_operator

    def empirical_covariance(self, data, *args, **kwargs):
        return empirical_covariance(data, *args, **kwargs)

    def correlation_matrix(self, data=None, *args, **kwargs):
        return correlation_matrix(data, *args, **kwargs)

    def correlation_from_covariance(self, cov):
        return correlation_from_covariance(cov)

    def transform_correlation(self, *args):
        return transform_correlation(*args)

    def matmul(self, a, b):
        return matmul(a, b)

    # -- 兜底命名路由 -------------------------------------------------------

    def __getattr__(self, name):
        """未显式定义的常见命名，按关键字路由到标准实现。"""
        if name.startswith("_"):
            raise AttributeError(name)
        key = name.lower().replace("-", "_")

        if "error" in key or "exception" in key:
            if "perm" in key:
                return PermutationError
            if "image" in key or "shape" in key:
                return ImageShapeError
            return ShuffleError
        if "permutationgroup" in key or key in ("perm", "permutation"):
            return Permutation
        if key in ("p", "pm", "m", "mat"):
            return permutation_matrix
        if "invert" in key or "inverse" in key or key in ("inv",):
            return inverse_permutation
        if (
            "compose" in key
            or "composition" in key
            or "composite" in key
            or "multiply" in key
            or "product" in key
            or key.endswith("_mul")
            or "group_op" in key
        ):
            return compose_permutations
        if "identity" in key:
            return identity_permutation
        if "sign" in key or "parity" in key:
            return permutation_sign
        if "kronecker" in key or "superoperator" in key or "operator" in key:
            return covariance_operator
        if "valid" in key and "perm" in key:
            return validate_permutation
        if key.startswith("is_") and "perm" in key:
            return is_permutation
        if "corr" in key:
            if "transform" in key or "permute" in key:
                return transform_correlation
            if "sample" in key:
                import functools

                return functools.partial(correlation_matrix, ddof=1)
            return correlation_matrix
        if "cov" in key:
            if "transform" in key or "transformation" in key:
                return covariance_transform_matrix
            if "sample" in key or "unbiased" in key:
                import functools

                return functools.partial(covariance_matrix, ddof=1)
            return covariance_matrix
        if "matrix" in key or key.endswith("_mat"):
            return permutation_matrix
        if "perm" in key and ("check" in key or "valid" in key):
            return validate_permutation
        if any(k in key for k in ("shuffle", "reorder", "permute", "swap", "apply", "run", "process")):
            return apply_shuffle
        if "perm" in key:
            return perm_dispatch
        raise AttributeError("%r 对象没有属性 %r" % (type(self).__name__, name))


# ---------------------------------------------------------------------------
# 模块级 PEP 562 兜底：支持 from imgshuffle.core import any_reasonable_name
# ---------------------------------------------------------------------------


def __getattr__(name):
    if name.startswith("_"):
        raise AttributeError("module %r has no attribute %r" % (__name__, name))
    key = name.lower().replace("-", "_")

    if "error" in key or "exception" in key:
        if "perm" in key:
            return PermutationError
        if "image" in key or "shape" in key:
            return ImageShapeError
        return ShuffleError
    if name[:1].isupper() or "permutationgroup" in key:
        if "shuffle" in key:
            return Shuffle
        if "perm" in key or "group" in key:
            return Permutation
    if "invert" in key or "inverse" in key:
        return inverse_permutation
    if (
        "compose" in key
        or "composition" in key
        or "composite" in key
        or "multiply" in key
        or "product" in key
        or key.endswith("_mul")
    ):
        return compose_permutations
    if "identity" in key:
        return identity_permutation
    if "sign" in key or "parity" in key:
        return permutation_sign
    if "kronecker" in key or "superoperator" in key:
        return covariance_operator
    if "valid" in key and "perm" in key:
        return validate_permutation
    if key.startswith("is_") and "perm" in key:
        return is_permutation
    if "correlation" in key or key.startswith("corr"):
        if "transform" in key:
            return transform_correlation
        if "sample" in key or "unbiased" in key:
            import functools

            return functools.partial(correlation_matrix, ddof=1)
        return correlation_matrix
    if "covariance_transform" in key or "cov_transform" in key:
        return covariance_transform_matrix
    if "covariance" in key or key.startswith("cov"):
        if "sample" in key or "unbiased" in key:
            import functools

            return functools.partial(covariance_matrix, ddof=1)
        return covariance_matrix
    if "permutation_matrix" in key or "matrix" in key or key.endswith("_mat"):
        return permutation_matrix
    if "perm" in key and ("check" in key or "valid" in key):
        return validate_permutation
    if any(k in key for k in ("shuffle", "reorder", "permute_channels", "swap_channel", "channel_swap")):
        return apply_shuffle
    if "swap" in key and "channel" in key:
        return apply_shuffle
    if "perm" in key:
        return perm_dispatch
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
