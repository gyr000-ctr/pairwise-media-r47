import math

import pytest

import imgglow

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False


def _norm(v):
    return math.sqrt(sum(c * c for c in v))

IMG = [[0.2, 0.9], [0.9, 0.2]]

def _g():
    return imgglow.Glow(0.5, 1.0)

CASES = [
    ("zero", lambda: all(abs(a - b) < 1e-12 for row_a, row_b in zip(_g().apply(IMG, 0.0), IMG) for a, b in zip(row_a, row_b))),
    ("range", lambda: all(0.0 <= v <= 1.0 for row in _g().apply(IMG, 1.0) for v in row)),
    ("threshold", lambda: isinstance(_g().threshold_map(IMG), list)),
    ("kernel_sum", lambda: abs(sum(_g().kernel(1.0)) - 1.0) < 1e-12),
    ("bright_added", lambda: _g().apply(IMG, 1.0)[0][1] >= IMG[0][1] - 1e-12),
    ("bad_sigma", lambda: _raises(lambda: imgglow.Glow(0.5, 0.0))),
    ("bad_threshold", lambda: _raises(lambda: imgglow.Glow(1.5, 1.0))),
    ("empty", lambda: _raises(lambda: _g().apply([], 1.0))),
    ("type", lambda: isinstance(_g().apply(IMG, 1.0), list)),
    ("det", lambda: _g().apply(IMG, 1.0) == _g().apply(IMG, 1.0)),
    ("nan", lambda: _raises(lambda: _g().apply([[float('nan')]], 1.0))),
    ("no_mut", lambda: _g().apply([r[:] for r in IMG], 1.0) is not None),
    ("kernel_type", lambda: isinstance(_g().kernel(1.0), list)),
    ("thresh_zero", lambda: all(abs(v) < 1e-12 for row in _g().threshold_map([[0.1, 0.2]]) for v in row)),
    ("float", lambda: isinstance(_g().apply(IMG, 1.0)[0][0], float)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
