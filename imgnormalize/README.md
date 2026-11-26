# imgnormalize

## 题面（可直接粘贴）

直方图归一（全局 + 分块 CLAHE 式）。约束：按分位拉伸且映射单调（≤1e-12），分位点精确映射到 0/1（≤1e-6）；必须实现局部分块拉伸，块边界双线性融合无接缝（相邻差有上界）；常量图不放大噪声（输出不变 ≤1e-12）；越界分位报错；100 万像素不超过 600 ms。 工作目录为仓库内的 imgnormalize/。 验收：cd imgnormalize && python -m pytest tests -q。 实现只允许使用 Python 标准库。

## 运行方式

```bash
cd imgnormalize && python -m pytest tests -q
```

初始 15 failed（实现前）。仅使用 Python 标准库。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图像处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
