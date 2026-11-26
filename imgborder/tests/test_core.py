import math

import pytest

import imgborder

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False


def _norm(v):
    return math.sqrt(sum(c * c for c in v))

IMG = [[1.0, 2.0], [3.0, 4.0]]

def _b():
    return imgborder.Border()

CASES = [
    ("size", lambda: len(_b().pad(IMG, 1, 1, 1, 1, 'zero')) == 4),
    ("zero", lambda: abs(_b().pad(IMG, 1, 1, 1, 1, 'zero')[0][0]) < 1e-12),
    ("replicate", lambda: abs(_b().pad(IMG, 1, 1, 1, 1, 'replicate')[0][0] - 1.0) < 1e-12),
    ("reflect", lambda: abs(_b().pad(IMG, 1, 1, 1, 1, 'reflect')[0][0] - 4.0) < 1e-12),
    ("wrap", lambda: abs(_b().pad(IMG, 1, 1, 1, 1, 'wrap')[0][0] - 4.0) < 1e-12),
    ("asym", lambda: len(_b().pad(IMG, 1, 0, 2, 0, 'zero')) == 3),
    ("bad_mode", lambda: _raises(lambda: _b().pad(IMG, 1, 1, 1, 1, 'bogus'))),
    ("bad_margin", lambda: _raises(lambda: _b().pad(IMG, -1, 1, 1, 1, 'zero'))),
    ("empty", lambda: _raises(lambda: _b().pad([], 1, 1, 1, 1, 'zero'))),
    ("type", lambda: isinstance(_b().pad(IMG, 1, 1, 1, 1, 'zero'), list)),
    ("det", lambda: _b().pad(IMG, 1, 1, 1, 1, 'zero') == _b().pad(IMG, 1, 1, 1, 1, 'zero')),
    ("nan", lambda: _raises(lambda: _b().pad([[float('nan')]], 1, 1, 1, 1, 'zero'))),
    ("no_mut", lambda: _b().pad([r[:] for r in IMG], 1, 1, 1, 1, 'zero') is not None),
    ("width", lambda: len(_b().pad(IMG, 0, 0, 1, 1, 'zero')[0]) == 4),
    ("float", lambda: isinstance(_b().pad(IMG, 1, 1, 1, 1, 'zero')[0][0], float)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
