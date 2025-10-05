import json
import shutil
from pathlib import Path

import pytest

from spaghettree.__main__ import main, run_process
from spaghettree.adapters.io_wrapper import FakeIOWrapper


@pytest.mark.parametrize(
    ("src_root"),
    [pytest.param("./mock_data/mock_case_1", id="Should run E2E without any errs")],
)
def test_e2e(src_root):
    try:
        tmp: Path = Path("./tmp_test").absolute()
        tmp.parent.mkdir(parents=True, exist_ok=True)
        res = main(src_root, new_root=f"{tmp}/src/case_1", optimise_src_code=True)
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
                    "from __future__ import annotations\n\n\n"
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
        pytest.param(
            "mock_data/mock_case_2/src/case_2",
            {
                "result/mock_data/mock_case_2/src/case_2/__init__.py": "from __future__ import annotations\n\nfrom case_2.mod_a import func_a, func_b\n"
                "from case_2.mod_a_isolated_func import isolated_func\nfrom case_2.mod_b import ClassA, func_c, func_d\n\n"
                '__all__: list[str] = ["ClassA", "func_a", "func_b", "func_c", "func_d", "isolated_func"]\n',
                "result/mock_data/mock_case_2/src/case_2/mod_a.py": "from __future__ import annotations\n\n\ndef func_a() -> int:\n"
                "    return 0 + func_b()\n"
                "\n"
                "\n"
                "def func_b() -> int:\n"
                "    return 1\n",
                "result/mock_data/mock_case_2/src/case_2/mod_a_isolated_func.py": "from __future__ import annotations\n\n\ndef isolated_func() -> int:\n"
                "    return 5\n",
                "result/mock_data/mock_case_2/src/case_2/mod_b.py": "from __future__ import annotations\n\n\ndef func_c() -> int:\n"
                "    return 2 + func_d()\n"
                "\n"
                "\n"
                "def func_d() -> int:\n"
                "    return 3\n"
                "\n"
                "\n"
                "class ClassA:\n"
                "    def method_a(self) -> int:\n"
                "        return func_d() + func_d()\n",
            },
            id="identify isolated function, handle ClassCST objects, retain `__all__`",
        ),
        pytest.param(
            "mock_data/mock_case_3/src/case_3",
            {
                "result/mock_data/mock_case_3/src/case_3/__init__.py": "",
                "result/mock_data/mock_case_3/src/case_3/mod_a.py": "from __future__ import annotations\n\nfrom case_3.mod_b import B\n"
                "\n"
                "\n"
                "class A:\n"
                "    pass\n"
                "\n"
                "\n"
                "C = A | B\n",
                "result/mock_data/mock_case_3/src/case_3/mod_a_mod_overflow.py": "from __future__ import annotations\n\nimport math\n"
                "\n"
                "\n"
                "def func_a() -> int:\n"
                "    return math.ceil(0.5)\n"
                "\n"
                "\n"
                "def func_b() -> int:\n"
                "    return func_a() + func_a()\n",
                "result/mock_data/mock_case_3/src/case_3/mod_b.py": "from __future__ import annotations\n\nCONSTANT = 3_000\n"
                "\n"
                "\n"
                "class B:\n"
                "    def method_a(self) -> int:\n"
                "        return CONSTANT\n",
            },
            id="ensure adds init and combines based on global typedef and all imports are correct",
        ),
        pytest.param(
            "mock_data/mock_case_4/src/case_4",
            {
                "result/mock_data/mock_case_4/src/case_4/__init__.py": "",
                "result/mock_data/mock_case_4/src/case_4/mod_a.py": "from __future__ import annotations\n\nimport math as mt\n\n\n"
                "def func_a() -> int:\n"
                "    return mt.ceil(CONSTANT)\n"
                "\n"
                "\n"
                "def func_b() -> int:\n"
                "    return func_a() + func_a()\n"
                "\n"
                "\n"
                "CONSTANT: float = 1_000.99\n",
                "result/mock_data/mock_case_4/src/case_4/mod_b.py": "from __future__ import annotations\n\nCONSTANT = 3_000\n"
                "\n"
                "\n"
                "class B:\n"
                "    def method_a(self) -> int:\n"
                "        return CONSTANT\n",
            },
            id="ensure doesn't confuse two constants",
        ),
        pytest.param(
            "mock_data/mock_case_5/src/case_5",
            {
                "result/mock_data/mock_case_5/src/case_5/__init__.py": "",
                "result/mock_data/mock_case_5/src/case_5/mod_a.py": "from __future__ import annotations\n\nimport attrs\n"
                "\n"
                "\n"
                "@attrs.define\n"
                "class SomeClass:\n"
                "    name: str = attrs.field()\n"
                "\n"
                "    def method_a(self) -> str:\n"
                "        return self.name.upper()\n",
            },
            id="ensure ignores empty module, ensure retains decorator or class level imports",
        ),
        pytest.param(
            "mock_data/mock_case_6/src/case_6",
            {
                "result/mock_data/mock_case_6/src/case_6/__init__.py": "",
                "result/mock_data/mock_case_6/src/case_6/case_6.py": "from __future__ import annotations\n\nimport math\n"
                "\n"
                "PI: float = math.pi\n"
                "\n"
                "\n"
                "class Circle:\n"
                "    def __init__(self, radius: float) -> None:\n"
                "        self.radius = radius\n"
                "\n"
                "    def calc_area(self) -> float:\n"
                "        return PI * self.radius * self.radius\n",
                "result/mock_data/mock_case_6/src/case_6/mod_b.py": "from __future__ import annotations\n\nfrom case_6.case_6 import PI\n"
                "\n"
                "\n"
                "def calculate_circumference(radius: float) -> float:\n"
                "    return 2 * PI * radius\n",
            },
            id="ensure combines modules when theres a global that imports from another package",
        ),
        pytest.param(
            "mock_data/mock_case_7/src/case_7",
            {
                "result/mock_data/mock_case_7/src/case_7/__init__.py": "",
                "result/mock_data/mock_case_7/src/case_7/case_7.py": "from __future__ import annotations\n\nfrom typing import Protocol, runtime_checkable\n"
                "\n"
                "\n"
                "@runtime_checkable\n"
                "class MockProtocol(Protocol):\n"
                "    def some_method(self, a: str) -> str: ...\n"
                "\n"
                "\n"
                "class FreeClass:\n"
                "    def some_other_method(self, a: int, b: int) -> int:\n"
                "        return a * b\n",
                "result/mock_data/mock_case_7/src/case_7/mod_protocol_mod_overflow.py": "from __future__ import annotations\n\nimport abc\n"
                "\n"
                "\n"
                "class Base(abc.ABC):\n"
                "    @abc.abstractmethod\n"
                "    def some_method(self, a: str) -> str:\n"
                "        pass\n"
                "\n"
                "\n"
                "class Child(Base):\n"
                "    def some_method(self, a: str) -> str:\n"
                "        return a.upper()\n",
            },
            id="ensure can capture inheritance hierarchies",
        ),
        pytest.param(
            "mock_data/mock_case_8/src/case_8",
            {
                "result/mock_data/mock_case_8/src/case_8/__init__.py": "",
                "result/mock_data/mock_case_8/src/case_8/case_8.py": "from __future__ import annotations\n\nfrom case_8.logger import logger\n"
                "\n"
                "\n"
                "def func_a() -> int:\n"
                '    logger.info("calling func_a")\n'
                "    return 0\n"
                "\n"
                "\n"
                "def func_b() -> int:\n"
                '    logger.debug("calling func b")\n'
                "    return func_a() + float(func_a())\n",
                "result/mock_data/mock_case_8/src/case_8/logger/__init__.py": "from __future__ import annotations\n\nimport logging\n"
                "\n"
                "logger = logging.getLogger(__name__)\n",
            },
            id="ensure it leaves a logger module alone",
        ),
        pytest.param(
            "mock_data/mock_case_9/src/case_9",
            {
                "result/mock_data/mock_case_9/src/case_9/__init__.py": "",
                "result/mock_data/mock_case_9/src/case_9/mod_generics.py": "from __future__ import annotations\n\nfrom typing import TypeVar\n"
                "\n"
                "from case_9.mod_utils import T\n"
                "\n"
                "\n"
                "def process_list(items: list[str]) -> list[str]:\n"
                "    return process_data(items)\n"
                "\n"
                "\n"
                "def create_mapping(keys: list[str], values: list[int]) -> dict[str, int]:\n"
                "    return dict(zip(keys, values, strict=False))\n"
                "\n"
                "\n"
                "def process_data[T](data: list[T]) -> list[T]:\n"
                "    return [item for item in data if item is not None]\n",
                "result/mock_data/mock_case_9/src/case_9/mod_generics_find_item.py": "from __future__ import annotations\n\n"
                "\n"
                "def find_item(items: list[str], target: str) -> str | None:\n"
                "    for item in items:\n"
                "        if item == target:\n"
                "            return item\n"
                "    return None\n",
                "result/mock_data/mock_case_9/src/case_9/mod_utils.py": "from __future__ import annotations\n\nfrom typing import TypeVar\n"
                "\n"
                'T = TypeVar("T")\n'
                "\n"
                "\n"
                "def get_first_item[T](items: list[T]) -> T:\n"
                "    return items[0]\n",
            },
            id="ensure handles generic types and TypeVar correctly",
        ),
    ],
    indirect=["fixture_get_subset_files"],
)
def test_run_process(fixture_get_subset_files, expected_result):
    name, files = fixture_get_subset_files
    io = FakeIOWrapper(files=files)
    run_process(io, name, new_root=f"result/{name}", optimise_src_code=True)
    assert {k: v for k, v in io.files.items() if k.startswith("result/")} == expected_result


@pytest.mark.parametrize(
    ("fixture_get_subset_files", "expected_result"),
    [
        pytest.param(
            "mock_data/mock_case_1/src/case_1",
            {
                "case_1.mod_a.func_a": [],
                "case_1.mod_b.func_b": [
                    "case_1.mod_a.func_a",
                    "case_1.mod_a.func_a",
                ],
            },
            id="ensure identifies the correct call tree for case_1",
        ),
        pytest.param(
            "mock_data/mock_case_2/src/case_2",
            {
                "case_2.__init__.__all__": [],
                "case_2.mod_a.func_a": [
                    "case_2.mod_a.func_b",
                ],
                "case_2.mod_a.func_b": [],
                "case_2.mod_a.func_c": [
                    "case_2.mod_b.func_d",
                ],
                "case_2.mod_a.isolated_func": [],
                "case_2.mod_b.ClassA": [
                    "case_2.mod_b.func_d",
                    "case_2.mod_b.func_d",
                ],
                "case_2.mod_b.func_d": [],
            },
            id="ensure identifies the correct call tree for case_2",
        ),
        pytest.param(
            "mock_data/mock_case_3/src/case_3",
            {
                "case_3.mod_a.A": [],
                "case_3.mod_a.func_a": [],
                "case_3.mod_a.func_b": [
                    "case_3.mod_a.func_a",
                    "case_3.mod_a.func_a",
                ],
                "case_3.mod_b.B": [
                    "case_3.mod_b.CONSTANT",
                ],
                "case_3.mod_b.C": [
                    "case_3.mod_a.A",
                    "case_3.mod_b.B",
                ],
                "case_3.mod_b.CONSTANT": [],
            },
            id="ensure identifies the correct call tree for case_3",
        ),
        pytest.param(
            "mock_data/mock_case_4/src/case_4",
            {
                "case_4.mod_a.CONSTANT": [],
                "case_4.mod_a.func_a": [
                    "case_4.mod_a.CONSTANT",
                ],
                "case_4.mod_a.func_b": [
                    "case_4.mod_a.func_a",
                    "case_4.mod_a.func_a",
                ],
                "case_4.mod_b.B": [
                    "case_4.mod_b.CONSTANT",
                ],
                "case_4.mod_b.CONSTANT": [],
            },
            id="ensure identifies the correct call tree for case_4",
        ),
        pytest.param(
            "mock_data/mock_case_5/src/case_5",
            {
                "case_5.mod_a.SomeClass": [],
            },
            id="ensure identifies the correct call tree for case_5",
        ),
        pytest.param(
            "mock_data/mock_case_6/src/case_6",
            {
                "case_6.mod_a.Circle": [
                    "case_6.mod_a.PI",
                ],
                "case_6.mod_a.PI": [],
                "case_6.mod_b.calculate_circumference": [
                    "case_6.mod_a.PI",
                ],
            },
            id="ensure identifies the correct call tree for case_6",
        ),
        pytest.param(
            "mock_data/mock_case_7/src/case_7",
            {
                "case_7.mod_protocol.MockProtocol": [],
                "case_7.mod_protocol.Base": [],
                "case_7.mod_protocol.Child": ["case_7.mod_protocol.Base"],
                "case_7.mod_protocol.FreeClass": [],
            },
            id="ensure identifies the correct call tree for case_7",
        ),
        pytest.param(
            "mock_data/mock_case_8/src/case_8",
            {
                "case_8.logger.__init__.logger": [],
                "case_8.mod_a.func_a": [],
                "case_8.mod_b.func_b": [
                    "case_8.mod_a.func_a",
                    "case_8.mod_a.func_a",
                ],
            },
            id="ensure identifies the correct call tree for case_8",
        ),
        pytest.param(
            "mock_data/mock_case_9/src/case_9",
            {
                "case_9.mod_generics.process_list": [
                    "case_9.mod_utils.process_data",
                ],
                "case_9.mod_generics.create_mapping": [],
                "case_9.mod_generics.find_item": [],
                "case_9.mod_utils.T": [],
                "case_9.mod_utils.process_data": [
                    "case_9.mod_utils.T",
                ],
                "case_9.mod_utils.get_first_item": [
                    "case_9.mod_utils.T",
                ],
            },
            id="ensure identifies the correct call tree for case_9 with generics",
        ),
    ],
    indirect=["fixture_get_subset_files"],
)
def test_run_process_return_call_tree(fixture_get_subset_files, expected_result):
    name, files = fixture_get_subset_files
    io = FakeIOWrapper(files=files)
    res = run_process(io, name, optimise_src_code=False)
    assert res.is_ok()
    assert json.loads(io.files[Path("./call_tree.json").absolute()]) == expected_result
