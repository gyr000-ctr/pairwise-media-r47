# imgborder

## 题面（可直接粘贴）

边界扩展与卷积等价性。约束：zero/replicate/reflect/wrap 四模式与逐像素参考一致（≤1e-12）；扩展尺寸 = 原尺寸 + 2×边距（逐轴校验），非对称边距可配；在扩展图上做卷积必须与核内直接越界取值的实现一致（≤1e-12）；反射模式的奇偶规则确定；100 万像素不超过 500 ms；负边距报错。 工作目录为仓库内的 imgborder/。 验收：cd imgborder && python -m pytest tests -q。 实现只允许使用 Python 标准库。

## 运行方式

```bash
cd imgborder && python -m pytest tests -q
```

初始 15 failed（实现前）。仅使用 Python 标准库。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图像处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
