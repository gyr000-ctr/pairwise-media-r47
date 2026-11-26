# imgshuffle

## 题面（可直接粘贴）

通道重排与置换群校验。约束：按排列重排与逐像素参考一致（≤1e-12）；排列必须是 0..n-1 的置换；恒等排列不变（≤1e-12）；逆排列往返 ≤1e-12；连续两次重排等于复合置换（群封闭性，≤1e-12）；必须给出置换的协方差变换矩阵并校验通道相关性变化与解析值一致（≤1e-9）；100 万像素不超过 400 ms。 工作目录为仓库内的 imgshuffle/。 验收：cd imgshuffle && python -m pytest tests -q。 实现只允许使用 Python 标准库。

## 运行方式

```bash
cd imgshuffle && python -m pytest tests -q
```

初始 15 failed（实现前）。仅使用 Python 标准库。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图像处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
