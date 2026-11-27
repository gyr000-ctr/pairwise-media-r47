from .core import *  # noqa: F401,F403
from . import core  # noqa: F401


def __getattr__(name):
    # 把未显式导出的近名属性委托给 core 的 PEP 562 解析器。
    return core.__getattr__(name)
