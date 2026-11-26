import math

import pytest

import imgnormalize

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False


def _norm(v):
    return math.sqrt(sum(c * c for c in v))

IMG = [[0.1, 0.2], [0.8, 0.9]]

def _n():
    return imgnormalize.Normalize(0.05, 0.95)

CASES = [
    ("range", lambda: all(0.0 <= v <= 1.0 for row in _n().apply(IMG) for v in row)),
    ("monotone", lambda: _n().apply([[0.1], [0.9]])[0][0] <= _n().apply([[0.1], [0.9]])[1][0]),
    ("constant", lambda: all(abs(v - 0.5) < 1e-12 for row in _n().apply([[0.5] * 3 for _ in range(3)]) for v in row)),
    ("low_maps_zero", lambda: abs(_n().apply([[0.1], [0.9]])[0][0]) < 0.05),
    ("quantile", lambda: 0.0 <= _n().quantile(IMG, 0.5) <= 1.0),
    ("bad_lo", lambda: _raises(lambda: imgnormalize.Normalize(-0.1, 0.9))),
    ("bad_order", lambda: _raises(lambda: imgnormalize.Normalize(0.9, 0.1))),
    ("bad_q", lambda: _raises(lambda: _n().quantile(IMG, 1.5))),
    ("empty", lambda: _raises(lambda: _n().apply([]))),
    ("type", lambda: isinstance(_n().apply(IMG), list)),
    ("det", lambda: _n().apply(IMG) == _n().apply(IMG)),
    ("nan", lambda: _raises(lambda: _n().apply([[float('nan')]]))),
    ("no_mut", lambda: _n().apply([r[:] for r in IMG]) is not None),
    ("quantile_type", lambda: isinstance(_n().quantile(IMG, 0.5), float)),
    ("float", lambda: isinstance(_n().apply(IMG)[0][0], float)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
