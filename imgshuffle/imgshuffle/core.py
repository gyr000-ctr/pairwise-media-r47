"""imgshuffle 核心 API（初始未实现，仅标准库）。"""


class Shuffle:
    def apply(self, image, perm):
        """重排"""
        raise NotImplementedError

    def inverse(self, perm):
        """逆排列"""
        raise NotImplementedError
