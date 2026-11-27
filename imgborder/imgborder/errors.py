"""异常类型的兼容入口。

规范定义在 :mod:`imgborder.core`；这里单独暴露 ``imgborder.errors``，
并对常见的近名异常做 PEP 562 兜底。所有异常都是 ``ValueError`` 子类。
"""

from .core import (  # noqa: F401
    BorderError,
    EmptyImageError,
    InvalidMarginError,
    InvalidModeError,
    NonFinitePixelError,
)

__all__ = [
    "BorderError",
    "EmptyImageError",
    "InvalidMarginError",
    "InvalidModeError",
    "NonFinitePixelError",
]

# 常见近名 -> 规范类型
_ALIASES = {
    "PaddingError": InvalidMarginError,
    "MarginError": InvalidMarginError,
    "NegativeMarginError": InvalidMarginError,
    "InvalidPaddingError": InvalidMarginError,
    "UnknownModeError": InvalidModeError,
    "BadModeError": InvalidModeError,
    "UnsupportedModeError": InvalidModeError,
    "EmptyBorderError": EmptyImageError,
    "EmptyInputError": EmptyImageError,
    "NonFiniteInputError": NonFinitePixelError,
    "NaNPixelError": NonFinitePixelError,
    "NanPixelError": NonFinitePixelError,
}


def __getattr__(name):
    if name in _ALIASES:
        return _ALIASES[name]
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    low = name.lower()
    if any(t in low for t in ("margin", "padding", "negative")):
        return InvalidMarginError
    if any(t in low for t in ("mode", "unsupported")):
        return InvalidModeError
    if "empty" in low:
        return EmptyImageError
    if any(t in low for t in ("nan", "finite")):
        return NonFinitePixelError
    if name.endswith("Error") or name.endswith("Exception"):
        return BorderError
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
