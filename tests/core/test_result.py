import pytest

from spaghettree.core.result import safe


def test_safe():
    @safe
    def raises():
        raise ValueError("An error")

    res = raises()
    assert not res.is_ok()
    assert list(res.details[0].keys()) == ["file", "func", "line_no", "locals"]
    # make sure Err.and_then => Err
    assert res.and_then(lambda x: x) == res

    with pytest.raises(ValueError):
        res.unwrap()
