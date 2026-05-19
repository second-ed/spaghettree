from typing import Any

import pytest
from libcst.metadata import CodePosition, CodeRange, QualifiedName, QualifiedNameSource

from spaghettree.core.logger import REPO_ROOT
from spaghettree.v2.domain.imports import ImportType
from spaghettree.v2.domain.parsing import collect_node_metadata
from spaghettree.v2.domain.utils import cst_to_str
from spaghettree.v2.domain.visitors import NodeMetadata


def _convert_node_to_comparable_obj(node: NodeMetadata) -> dict[str, Any]:
    items = node.to_dict()
    items["node"] = cst_to_str(node.node)
    items["scope"] = node.scope.__class__.__name__
    return items


@pytest.mark.parametrize(
    ("root", "expected_result"),
    [
        pytest.param(
            f"{REPO_ROOT}/mock_data/mock_case_1/src/case_1",
            [
                {
                    "calls": [],
                    "filepath": f"{REPO_ROOT}/mock_data/mock_case_1/src/case_1/mod_a.py",
                    "imports": [],
                    "node": "def func_a() -> int:\n    return 0\n",
                    "position": CodeRange(start=CodePosition(line=1, column=0), end=CodePosition(line=2, column=12)),
                    "qualified_names": (QualifiedName("mod_a.func_a", QualifiedNameSource.LOCAL),),
                    "references": [
                        QualifiedName("mod_a.func_a", QualifiedNameSource.LOCAL),
                        QualifiedName("builtins.int", QualifiedNameSource.BUILTIN),
                    ],
                    "scope": "GlobalScope",
                },
                {
                    "calls": [
                        QualifiedName("case_1.mod_a.func_a", QualifiedNameSource.IMPORT),
                        QualifiedName("builtins.float", QualifiedNameSource.BUILTIN),
                        QualifiedName("case_1.mod_a.func_a", QualifiedNameSource.IMPORT),
                    ],
                    "filepath": f"{REPO_ROOT}/mock_data/mock_case_1/src/case_1/mod_b.py",
                    "imports": [
                        {
                            "import_as_name": "func_a",
                            "import_module": "case_1.mod_a",
                            "import_name": "func_a",
                            "import_type": ImportType.FROM,
                        }
                    ],
                    "node": "\n\ndef func_b() -> int:\n    return func_a() + float(func_a())\n",
                    "position": CodeRange(start=CodePosition(line=4, column=0), end=CodePosition(line=5, column=37)),
                    "qualified_names": (QualifiedName("mod_b.func_b", QualifiedNameSource.LOCAL),),
                    "references": [
                        QualifiedName("mod_b.func_b", QualifiedNameSource.LOCAL),
                        QualifiedName("builtins.int", QualifiedNameSource.BUILTIN),
                        QualifiedName("case_1.mod_a.func_a", QualifiedNameSource.IMPORT),
                        QualifiedName("builtins.float", QualifiedNameSource.BUILTIN),
                        QualifiedName("case_1.mod_a.func_a", QualifiedNameSource.IMPORT),
                    ],
                    "scope": "GlobalScope",
                },
            ],
        ),
        pytest.param(
            f"{REPO_ROOT}/mock_data/mock_case_9/src/case_9",
            [
                {
                    "calls": [QualifiedName(name="case_9.mod_utils.process_data", source=QualifiedNameSource.IMPORT)],
                    "filepath": f"{REPO_ROOT}/mock_data/mock_case_9/src/case_9/mod_generics.py",
                    "imports": [
                        {
                            "import_as_name": "annotations",
                            "import_module": "__future__",
                            "import_name": "annotations",
                            "import_type": ImportType.FROM,
                        },
                        {
                            "import_as_name": "process_data",
                            "import_module": "case_9.mod_utils",
                            "import_name": "process_data",
                            "import_type": ImportType.FROM,
                        },
                    ],
                    "node": "\n\ndef process_list(items: list[str]) -> list[str]:\n    return process_data(items)\n",
                    "position": CodeRange(start=CodePosition(line=6, column=0), end=CodePosition(line=7, column=30)),
                    "qualified_names": (
                        QualifiedName(name="mod_generics.process_list", source=QualifiedNameSource.LOCAL),
                    ),
                    "references": [
                        QualifiedName(name="mod_generics.process_list", source=QualifiedNameSource.LOCAL),
                        QualifiedName(
                            name="mod_generics.process_list.<locals>.items", source=QualifiedNameSource.LOCAL
                        ),
                        QualifiedName(name="builtins.list", source=QualifiedNameSource.BUILTIN),
                        QualifiedName(name="builtins.str", source=QualifiedNameSource.BUILTIN),
                        QualifiedName(name="builtins.list", source=QualifiedNameSource.BUILTIN),
                        QualifiedName(name="builtins.str", source=QualifiedNameSource.BUILTIN),
                        QualifiedName(name="case_9.mod_utils.process_data", source=QualifiedNameSource.IMPORT),
                        QualifiedName(
                            name="mod_generics.process_list.<locals>.items", source=QualifiedNameSource.LOCAL
                        ),
                    ],
                    "scope": "GlobalScope",
                },
                {
                    "calls": [
                        QualifiedName(name="builtins.dict", source=QualifiedNameSource.BUILTIN),
                        QualifiedName(name="builtins.zip", source=QualifiedNameSource.BUILTIN),
                    ],
                    "filepath": f"{REPO_ROOT}/mock_data/mock_case_9/src/case_9/mod_generics.py",
                    "imports": [
                        {
                            "import_as_name": "annotations",
                            "import_module": "__future__",
                            "import_name": "annotations",
                            "import_type": ImportType.FROM,
                        },
                        {
                            "import_as_name": "process_data",
                            "import_module": "case_9.mod_utils",
                            "import_name": "process_data",
                            "import_type": ImportType.FROM,
                        },
                    ],
                    "node": "\n\ndef create_mapping(keys: list[str], values: list[int]) -> dict[str, int]:\n    return dict(zip(keys, values, strict=False))\n",
                    "position": CodeRange(start=CodePosition(line=10, column=0), end=CodePosition(line=11, column=48)),
                    "qualified_names": (
                        QualifiedName(name="mod_generics.create_mapping", source=QualifiedNameSource.LOCAL),
                    ),
                    "references": [
                        QualifiedName(name="mod_generics.create_mapping", source=QualifiedNameSource.LOCAL),
                        QualifiedName(
                            name="mod_generics.create_mapping.<locals>.keys", source=QualifiedNameSource.LOCAL
                        ),
                        QualifiedName(name="builtins.list", source=QualifiedNameSource.BUILTIN),
                        QualifiedName(name="builtins.str", source=QualifiedNameSource.BUILTIN),
                        QualifiedName(
                            name="mod_generics.create_mapping.<locals>.values", source=QualifiedNameSource.LOCAL
                        ),
                        QualifiedName(name="builtins.list", source=QualifiedNameSource.BUILTIN),
                        QualifiedName(name="builtins.int", source=QualifiedNameSource.BUILTIN),
                        QualifiedName(name="builtins.dict", source=QualifiedNameSource.BUILTIN),
                        QualifiedName(name="builtins.str", source=QualifiedNameSource.BUILTIN),
                        QualifiedName(name="builtins.int", source=QualifiedNameSource.BUILTIN),
                        QualifiedName(name="builtins.dict", source=QualifiedNameSource.BUILTIN),
                        QualifiedName(name="builtins.zip", source=QualifiedNameSource.BUILTIN),
                        QualifiedName(
                            name="mod_generics.create_mapping.<locals>.keys", source=QualifiedNameSource.LOCAL
                        ),
                        QualifiedName(
                            name="mod_generics.create_mapping.<locals>.values", source=QualifiedNameSource.LOCAL
                        ),
                        QualifiedName(name="builtins.False", source=QualifiedNameSource.BUILTIN),
                    ],
                    "scope": "GlobalScope",
                },
                {
                    "calls": [],
                    "filepath": f"{REPO_ROOT}/mock_data/mock_case_9/src/case_9/mod_generics.py",
                    "imports": [
                        {
                            "import_as_name": "annotations",
                            "import_module": "__future__",
                            "import_name": "annotations",
                            "import_type": ImportType.FROM,
                        },
                        {
                            "import_as_name": "process_data",
                            "import_module": "case_9.mod_utils",
                            "import_name": "process_data",
                            "import_type": ImportType.FROM,
                        },
                    ],
                    "node": "\n\ndef find_item(items: list[str], target: str) -> str | None:\n    for item in items:\n        if item == target:\n            return item\n    return None\n",
                    "position": CodeRange(start=CodePosition(line=14, column=0), end=CodePosition(line=18, column=15)),
                    "qualified_names": (
                        QualifiedName(name="mod_generics.find_item", source=QualifiedNameSource.LOCAL),
                    ),
                    "references": [
                        QualifiedName(name="mod_generics.find_item", source=QualifiedNameSource.LOCAL),
                        QualifiedName(name="mod_generics.find_item.<locals>.items", source=QualifiedNameSource.LOCAL),
                        QualifiedName(name="builtins.list", source=QualifiedNameSource.BUILTIN),
                        QualifiedName(name="builtins.str", source=QualifiedNameSource.BUILTIN),
                        QualifiedName(name="mod_generics.find_item.<locals>.target", source=QualifiedNameSource.LOCAL),
                        QualifiedName(name="builtins.str", source=QualifiedNameSource.BUILTIN),
                        QualifiedName(name="builtins.str", source=QualifiedNameSource.BUILTIN),
                        QualifiedName(name="builtins.None", source=QualifiedNameSource.BUILTIN),
                        QualifiedName(name="mod_generics.find_item.<locals>.item", source=QualifiedNameSource.LOCAL),
                        QualifiedName(name="mod_generics.find_item.<locals>.items", source=QualifiedNameSource.LOCAL),
                        QualifiedName(name="mod_generics.find_item.<locals>.item", source=QualifiedNameSource.LOCAL),
                        QualifiedName(name="mod_generics.find_item.<locals>.target", source=QualifiedNameSource.LOCAL),
                        QualifiedName(name="mod_generics.find_item.<locals>.item", source=QualifiedNameSource.LOCAL),
                        QualifiedName(name="builtins.None", source=QualifiedNameSource.BUILTIN),
                    ],
                    "scope": "GlobalScope",
                },
                {
                    "calls": [QualifiedName(name="typing.TypeVar", source=QualifiedNameSource.IMPORT)],
                    "filepath": f"{REPO_ROOT}/mock_data/mock_case_9/src/case_9/mod_utils.py",
                    "imports": [
                        {
                            "import_as_name": "TypeVar",
                            "import_module": "typing",
                            "import_name": "TypeVar",
                            "import_type": ImportType.FROM,
                        }
                    ],
                    "node": 'T = TypeVar("T")',
                    "position": CodeRange(start=CodePosition(line=3, column=0), end=CodePosition(line=3, column=16)),
                    "qualified_names": (QualifiedName(name="mod_utils.T", source=QualifiedNameSource.LOCAL),),
                    "references": [
                        QualifiedName(name="mod_utils.T", source=QualifiedNameSource.LOCAL),
                        QualifiedName(name="typing.TypeVar", source=QualifiedNameSource.IMPORT),
                    ],
                    "scope": "GlobalScope",
                },
                {
                    "calls": [],
                    "filepath": f"{REPO_ROOT}/mock_data/mock_case_9/src/case_9/mod_utils.py",
                    "imports": [
                        {
                            "import_as_name": "TypeVar",
                            "import_module": "typing",
                            "import_name": "TypeVar",
                            "import_type": ImportType.FROM,
                        }
                    ],
                    "node": "\n\ndef process_data[T](data: list[T]) -> list[T]:\n    return [item for item in data if item is not None]\n",
                    "position": CodeRange(start=CodePosition(line=6, column=0), end=CodePosition(line=7, column=54)),
                    "qualified_names": (
                        QualifiedName(name="mod_utils.process_data", source=QualifiedNameSource.LOCAL),
                    ),
                    "references": [
                        QualifiedName(name="mod_utils.process_data", source=QualifiedNameSource.LOCAL),
                        QualifiedName(name="mod_utils.process_data.<locals>.data", source=QualifiedNameSource.LOCAL),
                        QualifiedName(name="builtins.list", source=QualifiedNameSource.BUILTIN),
                        QualifiedName(name="mod_utils.T", source=QualifiedNameSource.LOCAL),
                        QualifiedName(name="builtins.list", source=QualifiedNameSource.BUILTIN),
                        QualifiedName(name="mod_utils.T", source=QualifiedNameSource.LOCAL),
                        QualifiedName(
                            name="mod_utils.process_data.<locals>.<comprehension>.item",
                            source=QualifiedNameSource.LOCAL,
                        ),
                        QualifiedName(
                            name="mod_utils.process_data.<locals>.<comprehension>.item",
                            source=QualifiedNameSource.LOCAL,
                        ),
                        QualifiedName(name="mod_utils.process_data.<locals>.data", source=QualifiedNameSource.LOCAL),
                        QualifiedName(
                            name="mod_utils.process_data.<locals>.<comprehension>.item",
                            source=QualifiedNameSource.LOCAL,
                        ),
                        QualifiedName(name="builtins.None", source=QualifiedNameSource.BUILTIN),
                    ],
                    "scope": "GlobalScope",
                },
                {
                    "calls": [],
                    "filepath": f"{REPO_ROOT}/mock_data/mock_case_9/src/case_9/mod_utils.py",
                    "imports": [
                        {
                            "import_as_name": "TypeVar",
                            "import_module": "typing",
                            "import_name": "TypeVar",
                            "import_type": ImportType.FROM,
                        }
                    ],
                    "node": "\n\ndef get_first_item[T](items: list[T]) -> T:\n    return items[0]\n",
                    "position": CodeRange(start=CodePosition(line=10, column=0), end=CodePosition(line=11, column=19)),
                    "qualified_names": (
                        QualifiedName(name="mod_utils.get_first_item", source=QualifiedNameSource.LOCAL),
                    ),
                    "references": [
                        QualifiedName(name="mod_utils.get_first_item", source=QualifiedNameSource.LOCAL),
                        QualifiedName(name="mod_utils.get_first_item.<locals>.items", source=QualifiedNameSource.LOCAL),
                        QualifiedName(name="builtins.list", source=QualifiedNameSource.BUILTIN),
                        QualifiedName(name="mod_utils.T", source=QualifiedNameSource.LOCAL),
                        QualifiedName(name="mod_utils.T", source=QualifiedNameSource.LOCAL),
                        QualifiedName(name="mod_utils.get_first_item.<locals>.items", source=QualifiedNameSource.LOCAL),
                    ],
                    "scope": "GlobalScope",
                },
            ],
        ),
    ],
)
def test_collect_node_metadata(root, expected_result) -> None:
    res = collect_node_metadata(root)
    assert res.is_ok()

    nodes = res.unwrap()
    assert list(map(_convert_node_to_comparable_obj, nodes)) == expected_result
