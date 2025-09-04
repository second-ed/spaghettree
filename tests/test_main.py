import os
import shutil
from pathlib import Path

import pytest

from spaghettree.__main__ import main
from spaghettree.adapters.io_wrapper import IOWrapper


@pytest.mark.parametrize(
    ("src_root", "expected_result"),
    [
        pytest.param(
            "./mock_package/src",
            {
                "./mock_package/src/mock_package/__init__.py": "",
                "./mock_package/src/mock_package/module_a.py": "def func_a():  # noqa: ANN201\n    return 1 + func_b()\n\n\ndef func_b():  # noqa: ANN201\n    return 1\n",
                "./mock_package/src/mock_package/module_a_isolated_func.py": "def isolated_func():  # noqa: ANN201\n    return 2\n",
                "./mock_package/src/mock_package/module_b.py": "def func_c():  # noqa: ANN201\n    return 2 * func_d()\n\n\ndef func_d():  # noqa: ANN201\n    return -2\n\n\nclass ClassA:\n    def method_a(self):  # noqa: ANN201\n        return func_d() + func_d()\n",
                "./mock_package/src/mock_package/module_b_mod_overflow.py": "from mock_package.module_b import func_d\n\nCONSTANT = 0\n\n\ndef func_e(a: int, b: int) -> int:\n    return a + b + func_d() + CONSTANT\n",
            },
        )
    ],
)
def test_main(src_root, expected_result):
    try:
        tmp = str(Path("./tmp_test_src_dir").absolute())
        os.makedirs(tmp, exist_ok=True)
        res = main(src_root, tmp)
        assert res.is_ok()

        io = IOWrapper()
        files_res = io.read_files(tmp)
        assert files_res.is_ok()

        files = files_res.inner
        # replace the garbage tmp_name with something deterministic
        files = {k.replace(tmp, src_root): v for k, v in files.items()}
        assert files == expected_result

    finally:
        shutil.rmtree(tmp)
