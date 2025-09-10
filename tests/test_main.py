import os
import shutil
from pathlib import Path

import pytest

from spaghettree.__main__ import main, run_process
from spaghettree.adapters.io_wrapper import FakeIOWrapper


@pytest.mark.parametrize(
    ("src_root"),
    [pytest.param("./mock_data/mock_case_1/src/case_1", id="Should run E2E without any errs")],
)
def test_main(src_root):
    try:
        tmp = str(Path("./tmp_test").absolute())
        os.makedirs(tmp, exist_ok=True)
        res = main(src_root, f"{tmp}/src/case_1")
        assert res.is_ok()

    finally:
        shutil.rmtree(tmp)


@pytest.mark.parametrize(
    ("fixture_get_subset_files", "expected_result"),
    [
        pytest.param(
            "mock_data/mock_case_1/src/case_1",
            {
                "result/mock_data/mock_case_1/src/case_1/__init__.py": "",
                "result/mock_data/mock_case_1/src/case_1/case_1.py": (
                    "def func_a() -> int:\n"
                    "    return 0\n"
                    "\n"
                    "\n"
                    "def func_b() -> int:\n"
                    "    return func_a() + float(func_a())\n"
                ),
            },
            id="given a simple connection B -> A, the modules should be combined",
        ),
        # pytest.param(
        #     "mock_data/mock_case_2/src/case_2",
        #     {
        #         "result/mock_data/mock_case_2/src/case_2/__init__.py": "",
        #         "result/mock_data/mock_case_2/src/case_2/mod_a.py": "from case_2.mod_b import func_d\n"
        #         "\n"
        #         "\n"
        #         "def func_a() -> int:\n"
        #         "    return 0 + func_b()\n"
        #         "\n"
        #         "\n"
        #         "def func_b() -> int:\n"
        #         "    return 1\n",
        #         "result/mock_data/mock_case_2/src/case_2/mod_a_isolated_func.py": "from case_2.mod_a import func_b\n"
        #         "from case_2.mod_b import func_d\n"
        #         "\n"
        #         "\n"
        #         "def isolated_func() -> int:\n"
        #         "    return 5\n",
        #         "result/mock_data/mock_case_2/src/case_2/mod_b.py": "from case_2.mod_a import func_b\n"
        #         "\n"
        #         "\n"
        #         "def func_c() -> int:\n"
        #         "    return 2 + func_d()\n"
        #         "\n"
        #         "\n"
        #         "def func_d() -> int:\n"
        #         "    return 3\n"
        #         "\n"
        #         "\n"
        #         "class ClassA:\n"
        #         "    def method_a(self) -> int:\n"
        #         "        return func_d() + func_d()\n",
        #     },
        #     id="identify isolated function, handle ClassCST objects",
        # ),
        # pytest.param(
        #     "mock_data/mock_case_3/src/case_3",
        #     {
        #         "result/mock_data/mock_case_3/src/case_3/__init__.py": "",
        #         "result/mock_data/mock_case_3/src/case_3/mod_a.py": (
        #             "from case_3.mod_b_a import A\n"
        #             "\n"
        #             "CONSTANT = 3_000\n"
        #             "\n"
        #             "\n"
        #             "class B:\n"
        #             "    def method_a(self) -> int:\n"
        #             "        return CONSTANT\n"
        #             "\n"
        #             "\n"
        #             "C = A | B\n"
        #         ),
        #         "result/mock_data/mock_case_3/src/case_3/mod_b.py": (
        #             "def func_a() -> int:\n"
        #             "    return 0\n"
        #             "\n"
        #             "\n"
        #             "def func_b() -> int:\n"
        #             "    return func_a() + func_a()\n"
        #         ),
        #         "result/mock_data/mock_case_3/src/case_3/mod_b_a.py": (
        #             "from case_3.mod_b import func_a\n\n\nclass A:"
        #         ),
        #     },
        #     id="ensure adds init and combines based on global typedef",
        # ),
    ],
    indirect=["fixture_get_subset_files"],
)
def test_run_process(fixture_get_subset_files, expected_result):
    name, files = fixture_get_subset_files
    io = FakeIOWrapper(files)
    run_process(io, name, f"result/{name}")
    assert {k: v for k, v in io.files.items() if k.startswith("result/")} == expected_result
