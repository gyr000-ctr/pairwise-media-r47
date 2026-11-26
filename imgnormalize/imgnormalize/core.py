"""imgnormalize 核心 API（初始未实现，仅标准库）。"""


class Normalize:
    def __init__(self, lo, hi):
        """构造"""
        raise NotImplementedError

    def apply(self, image):
        """拉伸"""
        raise NotImplementedError

    def quantile(self, image, q):
        """分位值"""
        raise NotImplementedError
