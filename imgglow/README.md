# imgglow

## 题面（可直接粘贴）

发光（阈值 + 模糊叠加）。约束：阈值提取只保留高亮（≤1e-12）；模糊为可分离高斯且核和为 1（≤1e-12）；合成按加法后钳制，原图细节不变（未超阈值像素差 ≤1e-12）；强度 0 时恒等；100 万像素不超过 700 ms；非法参数报错。 工作目录为仓库内的 imgglow/。 验收：cd imgglow && python -m pytest tests -q。 实现只允许使用 Python 标准库。

## 运行方式

```bash
cd imgglow && python -m pytest tests -q
```

初始 15 failed（实现前）。仅使用 Python 标准库。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图像处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
