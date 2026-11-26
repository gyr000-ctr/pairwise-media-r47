import math

import pytest

import imgshuffle

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False


def _norm(v):
    return math.sqrt(sum(c * c for c in v))

IMG = [[(1.0, 2.0, 3.0)]]

def _s():
    return imgshuffle.Shuffle()

CASES = [
    ("identity", lambda: _s().apply(IMG, [0, 1, 2]) == IMG),
    ("swap", lambda: abs(_s().apply(IMG, [2, 1, 0])[0][0][0] - 3.0) < 1e-12),
    ("inverse", lambda: _s().inverse([2, 0, 1]) == [1, 2, 0]),
    ("roundtrip", lambda: _s().apply(_s().apply(IMG, [2, 0, 1]), _s().inverse([2, 0, 1])) == IMG),
    ("bad_perm", lambda: _raises(lambda: _s().apply(IMG, [0, 0, 1]))),
    ("bad_len", lambda: _raises(lambda: _s().apply(IMG, [0, 1]))),
    ("bad_index", lambda: _raises(lambda: _s().apply(IMG, [0, 1, 5]))),
    ("empty", lambda: _raises(lambda: _s().apply([], [0, 1, 2]))),
    ("type", lambda: isinstance(_s().apply(IMG, [1, 2, 0]), list)),
    ("det", lambda: _s().apply(IMG, [1, 2, 0]) == _s().apply(IMG, [1, 2, 0])),
    ("nan", lambda: _raises(lambda: _s().apply([[(float('nan'), 0.0, 0.0)]], [0, 1, 2]))),
    ("no_mut", lambda: _s().apply([list(r) for r in IMG], [0, 2, 1]) is not None),
    ("inv_type", lambda: isinstance(_s().inverse([1, 0, 2]), list)),
    ("perm_int", lambda: all(isinstance(i, int) for i in _s().inverse([1, 0, 2]))),
    ("float", lambda: isinstance(_s().apply(IMG, [1, 2, 0])[0][0][0], float)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
