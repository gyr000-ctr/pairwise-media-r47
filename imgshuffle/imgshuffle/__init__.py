from .core import *  # noqa: F401,F403
from . import core  # noqa: F401

try:
    from .core import __all__ as _core_all

    __all__ = list(_core_all) + ["core"]
except ImportError:  # pragma: no cover
    __all__ = ["core"]


def __getattr__(name):
    # 兜底：把未在 __all__ 中显式导出的合理命名委托给 core 的 PEP 562 路由。
    return getattr(core, name)
