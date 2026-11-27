# imgnormalize

## 题面（可直接粘贴）

直方图归一（全局 + 分块 CLAHE 式）。约束：按分位拉伸且映射单调（≤1e-12），分位点精确映射到 0/1（≤1e-6）；必须实现局部分块拉伸，块边界双线性融合无接缝（相邻差有上界）；常量图不放大噪声（输出不变 ≤1e-12）；越界分位报错；100 万像素不超过 600 ms。 工作目录为仓库内的 imgnormalize/。 验收：cd imgnormalize && python -m pytest tests -q。 实现只允许使用 Python 标准库。

## 运行方式

```bash
cd imgnormalize && python -m pytest tests -q
```

初始 15 failed（实现前）。仅使用 Python 标准库。

## 实现说明

API：`imgnormalize.Normalize(lo, hi, ...)` 提供 `apply` / `quantile`，模块级
`normalize` / `stretch` / `apply_local` / `clahe` / `quantile` 为便捷包装。
输入/输出统一为 `[[v, ...], ...]` 嵌套列表（也接受元组、一维列表与 NumPy 数组），
输出为 float 嵌套列表。

- **全局分位拉伸**：`y = clip((v - q_lo) / (q_hi - q_lo), 0, 1)`，分位点用线性
  插值（与 `numpy.percentile` 默认 "linear" 一致）。减法与正除数保持 IEEE 序关系，
  因此映射单调不减（相邻差 0）；`a / (b - a) == 0`、`(b - a) / (b - a) == 1`
  精确成立，两个分位点映射到 0 / 1 无误差。
- **局部分块（CLAHE 式）**：`apply(img, blocks=8)`（或 `tiles` / `grid` /
  `block_size`）把图像切成均衡分块，每块独立求 `lo`/`hi` 并做同一线性拉伸；
  像素值由包围其的至多 4 个块中心做双线性加权融合。权重只由像素坐标决定、
  随位置连续变化，块边界处没有跳变（相邻差有上界），输出始终在 `[0, 1]`。
  `blocks=1` 与全局模式逐元素一致（≤1e-16）。
- **常量保护**：块或整图的 `q_hi == q_lo` 时退化为恒等映射；整幅常量图在
  分块模式下直接返回原图，因此常量图逐像素原样输出（误差 0，不放大噪声）。
- **错误分类**：全部继承 `ValueError`：`NormalizeError`（基类）、
  `NormalizeTypeError`（类型/非数值，同时是 `TypeError`）、
  `NormalizeShapeError`（空图/空行/参差）、`NormalizeValueError`（数值），
  其下再细分 `NaNPixelError` / `InfinitePixelError`（非有限像素）、
  `InvalidQuantileError`（越界分位）、`InvalidParameterError`（lo/hi、块参数、
  mode 非法）。
- **不变量**：不修改输入、无全局可变状态（实例可多线程共享）、纯函数式结果确定。
- **性能**：100 万像素（1000×1000）全局约 230–280 ms，分块 8×8 约 270–340 ms
  （纯标准库；排序 `Timsort` 占主要开销，输出逐行装配避免逐像素解释器循环）。
  `tests/test_spec_acceptance.py` 另外覆盖题面全部约束（115 条回归用例）。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图像处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
