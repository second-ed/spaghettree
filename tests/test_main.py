import tempfile

from spaghettree.__main__ import main


def test_main():
    with tempfile.TemporaryDirectory() as tmp:
        res = main("./mock_package/src/mock_package", tmp)
        assert res.is_ok()
