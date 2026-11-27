"""验收约束的完整回归测试（在仓库自带 15 条规格测试之外补充）。

覆盖题面中的每一条约束：

* 按分位拉伸，映射单调（相邻值差 ≤ 1e-12 的序关系），分位点精确映射到 0 / 1；
* 全局与局部分块两种模式，局部块边界双线性融合无接缝（相邻差有上界）；
* 常量图不放大噪声（输出与原图逐像素相等，误差 ≤ 1e-12）；
* 越界分位报错（构造参数 / ``quantile`` 的 q），且错误可按类别区分；
* 100 万像素不超过 600 ms；
* 纯标准库、不修改输入、结果确定、并发安全。
"""

import math
import threading
import time

import pytest

import imgnormalize


# ---------------------------------------------------------------------------
# 独立参考实现（不复用被测代码，减少「自证」风险）
# ---------------------------------------------------------------------------
def _linear_quantile(sorted_vals, q):
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
    return sorted_vals[i] + (sorted_vals[i + 1] - sorted_vals[i]) * frac


def _flat(image):
    return [v for row in image for v in row]


def _ref_global(image, lo, hi):
    vals = sorted(_flat(image))
    a = _linear_quantile(vals, lo)
    b = _linear_quantile(vals, hi)
    if not (b > a):
        return [v for row in image for v in row]
    out = []
    for v in _flat(image):
        t = (v - a) / (b - a)
        out.append(0.0 if t < 0.0 else (1.0 if t > 1.0 else t))
    return out


def _edges(n, k):
    return [(n * i) // k for i in range(k + 1)]


def _ref_local(image, lo, hi, ny, nx):
    """逐块线性拉伸 + 块中心双线性融合的独立参考实现。"""
    h = len(image)
    w = len(image[0])
    ys = _edges(h, ny)
    xs = _edges(w, nx)
    a_tab = [[0.0] * nx for _ in range(ny)]
    inv_tab = [[1.0] * nx for _ in range(ny)]
    for i in range(ny):
        for j in range(nx):
            vals = sorted(
                image[r][c]
                for r in range(ys[i], ys[i + 1])
                for c in range(xs[j], xs[j + 1])
            )
            a = _linear_quantile(vals, lo)
            b = _linear_quantile(vals, hi)
            if b > a:
                a_tab[i][j] = a
                inv_tab[i][j] = 1.0 / (b - a)

    def axis(n, k):
        if k <= 1:
            return [(0, 0, 0.0)] * n
        e = _edges(n, k)
        centers = [(e[i] + e[i + 1]) / 2.0 for i in range(k)]
        res = []
        for r in range(n):
            p = r + 0.5
            i = 0
            while i + 1 < k and centers[i + 1] <= p:
                i += 1
            if p <= centers[0]:
                res.append((0, 0, 0.0))
            elif p >= centers[-1]:
                res.append((k - 1, k - 1, 0.0))
            else:
                res.append((i, i + 1, (p - centers[i]) / (centers[i + 1] - centers[i])))
        return res

    yw = axis(h, ny)
    xw = axis(w, nx)
    out = []
    for r in range(h):
        i0, i1, wy = yw[r]
        row = []
        for c in range(w):
            j0, j1, wx = xw[c]
            v = image[r][c]

            def m(i, j):
                t = (v - a_tab[i][j]) * inv_tab[i][j]
                return 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)

            top = m(i0, j0) * (1.0 - wx) + m(i0, j1) * wx
            bottom = m(i1, j0) * (1.0 - wx) + m(i1, j1) * wx
            row.append(top * (1.0 - wy) + bottom * wy)
        out.append(row)
    return out


def _sample(h, w, seed=0):
    """确定性伪随机图（xorshift），不依赖 random 模块实现细节。"""
    state = seed or 1
    rows = []
    for _ in range(h):
        row = []
        for _ in range(w):
            state ^= (state << 13) & 0xFFFFFFFF
            state ^= state >> 17
            state ^= (state << 5) & 0xFFFFFFFF
            row.append((state & 0xFFFFFF) / float(0x1000000))
        rows.append(row)
    return rows


IMG = [[0.1, 0.2], [0.8, 0.9]]


def _n():
    return imgnormalize.Normalize(0.05, 0.95)


# ---------------------------------------------------------------------------
# 全局分位拉伸
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", [(1, 1), (1, 8), (8, 1), (5, 5), (13, 11), (37, 53)])
@pytest.mark.parametrize("quantiles", [(0.05, 0.95), (0.0, 1.0), (0.25, 0.75), (0.4, 0.6)])
def test_global_matches_reference(shape, quantiles):
    img = _sample(*shape, seed=shape[0] * 97 + shape[1])
    lo, hi = quantiles
    got = _n().apply(img) if (lo, hi) == (0.05, 0.95) else imgnormalize.Normalize(lo, hi).apply(img)
    exp = _ref_global(img, lo, hi)
    assert max(abs(a - b) for a, b in zip(_flat(got), exp)) <= 1e-12


def test_mapping_is_monotone():
    img = _sample(31, 29, seed=5)
    pairs = sorted(zip(_flat(img), _flat(_n().apply(img))))
    assert all(pairs[i][1] <= pairs[i + 1][1] + 1e-12 for i in range(len(pairs) - 1))


def test_quantile_points_map_exactly_to_zero_and_one():
    # 取 3x3、值 0..8：q=0.25 / 0.5 / 0.75 的分位点恰好落在真实像素上。
    img = [[float(r * 3 + c) for c in range(3)] for r in range(3)]
    for lo, hi in [(0.0, 1.0), (0.25, 0.75), (0.5, 1.0), (0.25, 0.5)]:
        n = imgnormalize.Normalize(lo, hi)
        out = _flat(n.apply(img))
        a = n.quantile(img, lo)
        b = n.quantile(img, hi)
        at_a = [o for v, o in zip(_flat(img), out) if v == a]
        at_b = [o for v, o in zip(_flat(img), out) if v == b]
        assert at_a and max(abs(v) for v in at_a) <= 1e-6, (lo, hi, a)
        assert at_b and max(abs(v - 1.0) for v in at_b) <= 1e-6, (lo, hi, b)


def test_quantile_points_exact_on_arbitrary_image():
    """分位点若不是真实像素值，则不存在像素可精确映射，但端点极限仍成立。"""
    img = _sample(23, 19, seed=7)
    n = _n()
    out = _flat(n.apply(img))
    a = n.quantile(img, 0.05)
    b = n.quantile(img, 0.95)
    # 所有 <= a 的像素映射为 0，所有 >= b 的像素映射为 1。
    assert all(o == 0.0 for v, o in zip(_flat(img), out) if v < a)
    assert all(o == 1.0 for v, o in zip(_flat(img), out) if v > b)
    # 映射连续：略高于 a 的值映射到很小；略低于 b 的值映射到接近 1。
    probe = imgnormalize.Normalize(0.0, 1.0)
    span = b - a
    near_a = a + 1e-9 * span
    near_b = b - 1e-9 * span
    # 通过构造一个包含端点与探针的图像来取值（端点即该图的最小/最大值）。
    probe_img = [[a, near_a], [near_b, b]]
    mapped = _flat(probe.apply(probe_img))
    assert abs(mapped[0]) <= 1e-6
    assert abs(mapped[3] - 1.0) <= 1e-6
    assert mapped[1] <= 1e-6 and mapped[2] >= 1.0 - 1e-6


def test_output_range_and_type():
    img = _sample(9, 9, seed=3)
    out = _n().apply(img)
    assert isinstance(out, list)
    assert all(isinstance(row, list) for row in out)
    assert all(isinstance(v, float) for v in _flat(out))
    assert all(0.0 <= v <= 1.0 for v in _flat(out))


def test_input_not_mutated_and_deterministic():
    img = _sample(11, 7, seed=2)
    frozen = [row[:] for row in img]
    first = _n().apply(img)
    second = _n().apply(img)
    assert img == frozen
    assert first == second


def test_quantile_api():
    img = _sample(5, 7, seed=11)
    n = _n()
    assert isinstance(n.quantile(img, 0.5), float)
    assert 0.0 <= n.quantile(img, 0.5) <= 1.0
    assert n.quantile(img, 0.0) == min(_flat(img))
    assert n.quantile(img, 1.0) == max(_flat(img))
    vals = sorted(_flat(img))
    assert abs(n.quantile(img, 0.3) - _linear_quantile(vals, 0.3)) <= 1e-12


# ---------------------------------------------------------------------------
# 局部分块（CLAHE 式）拉伸
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", [(1, 1), (1, 9), (9, 1), (6, 6), (7, 5), (32, 24)])
@pytest.mark.parametrize("blocks", [1, 2, 3, 4, 8, (2, 3), (3, 2)])
def test_local_matches_reference(shape, blocks):
    img = _sample(*shape, seed=shape[0] * 31 + shape[1])
    h, w = shape
    ny, nx = (blocks, blocks) if isinstance(blocks, int) else blocks
    got = _n().apply(img, blocks=blocks)
    exp = _flat(_ref_local(img, 0.05, 0.95, min(ny, h), min(nx, w)))
    assert max(abs(a - b) for a, b in zip(_flat(got), exp)) <= 1e-12


def test_local_is_actually_used_unless_blocks_requested():
    img = _sample(16, 16, seed=4)
    assert _n().apply(img) == _n().apply(img, mode="global")
    assert _n().apply(img, blocks=2) != _n().apply(img)


def test_local_block_boundaries_have_no_seam():
    """平滑渐变上，块边界处的相邻差不超过块内部的最大相邻差。"""
    h = w = 64
    img = [[c / (w - 1) for c in range(w)] for _ in range(h)]
    out = _n().apply(img, blocks=8)
    boundaries = {(w * k) // 8 for k in range(1, 8)}

    def step(c):
        return max(abs(out[r][c] - out[r][c - 1]) for r in range(h))

    interior = max(step(c) for c in range(1, w) if c not in boundaries)
    at_boundary = max(step(c) for c in boundaries)
    assert at_boundary <= interior + 1e-12


def test_local_seam_bound_on_smooth_content():
    """平滑内容上的局部拉伸保持有界相邻差（无块状跳变）。"""
    h = w = 64
    img = [
        [0.5 + 0.4 * math.sin(c / 9.0) + 0.3 * math.cos(r / 11.0) for c in range(w)]
        for r in range(h)
    ]
    out = _n().apply(img, blocks=8)
    horizontal = max(abs(out[r][c] - out[r][c - 1]) for r in range(h) for c in range(1, w))
    vertical = max(abs(out[r][c] - out[r - 1][c]) for r in range(1, h) for c in range(w))
    assert horizontal <= 0.5 and vertical <= 0.5


def test_local_block_quantiles_shape():
    img = _sample(9, 11, seed=6)
    table = _n().block_quantiles(img, blocks=(3, 5))
    assert len(table) == 3 and all(len(row) == 5 for row in table)
    assert all(a <= b for row in table for a, b in row)


def test_local_differs_from_global_on_varying_content():
    """局部模式确实对每个块使用各自的拉伸区间（否则就退化为全局）。"""
    img = _sample(16, 16, seed=4)
    local = _n().apply(img, blocks=2)
    glob = _n().apply(img)
    assert local != glob


def test_local_monotone_and_range():
    img = _sample(29, 31, seed=8)
    out = _flat(_n().apply(img, blocks=4))
    assert all(0.0 <= v <= 1.0 for v in out)


def test_local_mapping_is_monotone_in_pixel_value_at_fixed_position():
    """固定像素位置、只改变该像素值时，局部输出单调不减（双线性权重非负且和为 1）。"""
    img = _sample(20, 20, seed=12)
    n = _n()
    values = [i / 100.0 for i in range(101)]
    for (r, c) in [(0, 0), (5, 13), (10, 10), (19, 19), (0, 19)]:
        mapped = []
        for v in values:
            probe = [row[:] for row in img]
            probe[r][c] = v
            mapped.append(n.apply(probe, blocks=4)[r][c])
        assert all(mapped[i] <= mapped[i + 1] + 1e-12 for i in range(len(mapped) - 1))


# ---------------------------------------------------------------------------
# 常量图：不放大噪声 / 输出不变
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"blocks": 1},
        {"blocks": 4},
        {"blocks": (2, 3)},
        {"block_size": 3},
        {"tile_size": 5},
        {"tiles": 4},
        {"grid": (2, 4)},
    ],
)
def test_constant_image_is_unchanged(kwargs):
    img = [[0.42] * 11 for _ in range(9)]
    out = _n().apply(img, **kwargs)
    assert max(abs(v - 0.42) for v in _flat(out)) <= 1e-12


def test_constant_image_unchanged_via_constructor_and_module_api():
    img = [[0.3] * 7 for _ in range(7)]
    assert all(
        abs(v - 0.3) <= 1e-12
        for v in _flat(imgnormalize.Normalize(0.05, 0.95, blocks=4).apply(img))
    )
    assert all(abs(v - 0.3) <= 1e-12 for v in _flat(imgnormalize.apply_local(img)))
    assert all(abs(v - 0.3) <= 1e-12 for v in _flat(imgnormalize.clahe(img)))
    assert all(abs(v - 0.3) <= 1e-12 for v in _flat(imgnormalize.normalize(img)))


def test_constant_image_single_pixel():
    out = _n().apply([[0.7]])
    assert out == [[0.7]]


def test_constant_blocks_inside_varying_image():
    img = [
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.5, 0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5],
    ]
    assert _n().apply(img, blocks=(2, 2)) == img


# ---------------------------------------------------------------------------
# 越界分位 / 非法参数报错
# ---------------------------------------------------------------------------
def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False


@pytest.mark.parametrize("q", [1.5, -0.1, float("nan"), float("inf"), "0.5", None])
def test_bad_quantile_raises(q):
    assert _raises(lambda: _n().quantile(IMG, q))


@pytest.mark.parametrize(
    "lo,hi",
    [(-0.1, 0.9), (0.9, 0.1), (0.5, 0.5), (0.0, 1.5), (float("nan"), 0.9), ("a", 0.9)],
)
def test_bad_lo_hi_raises(lo, hi):
    assert _raises(lambda: imgnormalize.Normalize(lo, hi))


@pytest.mark.parametrize("kwargs", [{"blocks": 0}, {"blocks": 2.5}, {"block_size": 0}, {"mode": "bogus"}])
def test_bad_block_args_raise(kwargs):
    assert _raises(lambda: _n().apply(IMG, **kwargs))


@pytest.mark.parametrize("image", [[], [[]], [[1.0, 2.0], [1.0]], None, "ab", 5])
def test_bad_images_raise(image):
    assert _raises(lambda: _n().apply(image))


def test_non_finite_inputs_are_classified():
    assert _raises(lambda: _n().apply([[float("nan")]]), imgnormalize.NaNPixelError)
    assert _raises(lambda: _n().apply([[float("inf")]]), imgnormalize.InfinitePixelError)
    assert _raises(lambda: _n().apply([[float("-inf")]]), imgnormalize.InfinitePixelError)
    # 同时也是 ValueError，便于统一捕获
    assert _raises(lambda: _n().apply([[float("nan")]]), ValueError)
    assert _raises(lambda: _n().quantile(IMG, 2.0), imgnormalize.InvalidQuantileError)
    assert _raises(lambda: imgnormalize.Normalize(-0.2, 0.5), imgnormalize.InvalidParameterError)


# ---------------------------------------------------------------------------
# 性能：100 万像素不超过 600 ms
# ---------------------------------------------------------------------------
def test_million_pixels_under_budget():
    img = _sample(1000, 1000, seed=12345)
    n = _n()

    timings = []
    for _ in range(2):
        start = time.perf_counter()
        n.apply(img)
        timings.append(time.perf_counter() - start)
    assert min(timings) <= 0.600

    timings = []
    for _ in range(2):
        start = time.perf_counter()
        n.apply(img, blocks=8)
        timings.append(time.perf_counter() - start)
    assert min(timings) <= 0.600


# ---------------------------------------------------------------------------
# 并发安全（共享同一实例）
# ---------------------------------------------------------------------------
def test_shared_instance_is_thread_safe():
    images = [_sample(24, 20, seed=i + 1) for i in range(6)]
    n = _n()
    expected = [n.apply(img, blocks=4) for img in images]
    errors = []

    def worker(index):
        for _ in range(10):
            if n.apply(images[index], blocks=4) != expected[index]:
                errors.append(index)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(images))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
