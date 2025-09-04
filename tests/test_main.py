import tempfile

import pytest

from spaghettree.__main__ import main, run_process
from spaghettree.adapters.io_wrapper import FakeIOWrapper


def test_main():
    with tempfile.TemporaryDirectory() as tmp:
        res = main("./mock_package/src/mock_package", tmp)
        assert res.is_ok()


STARTING_FILES = {
    "some/path/to/mock_package/src/mock_package/module_a.py": (
        "from mock_package.module_b import func_d  # noqa: INP001\n"
        "\n"
        "\n"
        "def func_a():  # noqa: ANN201\n"
        "    return 1 + func_b()\n"
        "\n"
        "\n"
        "def func_b():  # noqa: ANN201\n"
        "    return 1\n"
        "\n"
        "\n"
        "def func_c():  # noqa: ANN201\n"
        "    return 2 * func_d()\n"
        "\n"
        "\n"
        "def isolated_func():  # noqa: ANN201\n"
        "    return 2\n"
    ),
    "some/path/to/mock_package/src/mock_package/module_b.py": (
        "def func_e(a: int, b: int) -> int:  # noqa: INP001\n"
        "    return a + b + func_d()\n"
        "\n"
        "\n"
        "def func_d():  # noqa: ANN201\n"
        "    return -2\n"
        "\n"
        "\n"
        "class ClassA:\n"
        "    def method_a(self):  # noqa: ANN201\n"
        "        return func_d() + func_d()\n"
        "\n"
    ),
}


@pytest.mark.parametrize(
    ("files", "src_root", "expected_result"),
    [
        pytest.param(
            STARTING_FILES,
            "some/path/to/mock_package/src/mock_package",
            {
                "some/path/to/mock_package/src/mock_package/__init__.py": "",
                "some/path/to/mock_package/src/mock_package/module_a.py": (
                    "from mock_package.module_b import func_d\n"
                    "\n"
                    "\n"
                    "def func_a():  # noqa: ANN201\n"
                    "    return 1 + func_b()\n"
                    "\n"
                    "\n"
                    "def func_b():  # noqa: ANN201\n"
                    "    return 1\n"
                ),
                "some/path/to/mock_package/src/mock_package/module_a_isolated_func.py": (
                    "from mock_package.module_a import func_b\n"
                    "from mock_package.module_b import func_d\n"
                    "\n"
                    "\n"
                    "def isolated_func():  # noqa: ANN201\n"
                    "    return 2\n"
                ),
                "some/path/to/mock_package/src/mock_package/module_b.py": (
                    "from mock_package.module_a import func_b\n"
                    "\n"
                    "\n"
                    "def func_c():  # noqa: ANN201\n"
                    "    return 2 * func_d()\n"
                    "\n"
                    "\n"
                    "def func_e(a: int, b: int) -> int:  # noqa: INP001\n"
                    "    return a + b + func_d()\n"
                    "\n"
                    "\n"
                    "def func_d():  # noqa: ANN201\n"
                    "    return -2\n"
                    "\n"
                    "\n"
                    "class ClassA:\n"
                    "    def method_a(self):  # noqa: ANN201\n"
                    "        return func_d() + func_d()\n"
                ),
            },
        ),
    ],
)
def test_run_process(files, src_root, expected_result):
    io = FakeIOWrapper(files)
    run_process(io, src_root, new_root=None)
    assert io.files == expected_result
