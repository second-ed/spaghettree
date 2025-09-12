from spaghettree import safe


def test_safe():
    @safe
    def raises():
        raise ValueError("An error")

    res = raises()
    assert not res.is_ok()
    assert list(res.details[0].keys()) == ["file", "func", "line_no", "locals"]
