import pytest

from buggy import scale


def test_scale_basic():
    assert scale([1, 2], 2) == [2, 4]


def test_scale_empty_raises():
    with pytest.raises(ValueError, match="no values"):
        scale([], 2)
