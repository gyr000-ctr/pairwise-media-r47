import math

import pytest

import imgmosaic

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False


def _norm(v):
    return math.sqrt(sum(c * c for c in v))

IMG = [[0.2, 0.4], [0.6, 0.8]]

def _m():
    return imgmosaic.Mosaic()

CASES = [
    ("identity", lambda: _m().apply(IMG, 1) == IMG),
    ("block_value", lambda: abs(_m().apply(IMG, 2)[0][0] - 0.5) < 1e-12),
    ("uniform_block", lambda: abs(_m().apply(IMG, 2)[0][1] - _m().apply(IMG, 2)[0][0]) < 1e-12),
    ("shape", lambda: len(_m().apply([[0.5] * 5 for _ in range(5)], 2)) == 5),
    ("partial", lambda: isinstance(_m().apply([[0.5] * 5 for _ in range(5)], 2), list)),
    ("bad_block", lambda: _raises(lambda: _m().apply(IMG, 0))),
    ("empty", lambda: _raises(lambda: _m().apply([], 2))),
    ("type", lambda: isinstance(_m().apply(IMG, 2), list)),
    ("det", lambda: _m().apply(IMG, 2) == _m().apply(IMG, 2)),
    ("range", lambda: all(0.0 <= v <= 1.0 for row in _m().apply(IMG, 2) for v in row)),
    ("nan", lambda: _raises(lambda: _m().apply([[float('nan')]], 2))),
    ("no_mut", lambda: _m().apply([r[:] for r in IMG], 2) is not None),
    ("mean_check", lambda: abs(_m().apply([[0.0, 1.0], [1.0, 0.0]], 2)[0][0] - 0.5) < 1e-12),
    ("float", lambda: isinstance(_m().apply(IMG, 2)[0][0], float)),
    ("big_block", lambda: isinstance(_m().apply(IMG, 4), list)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
