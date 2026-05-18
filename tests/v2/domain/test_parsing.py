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
        )
    ],
)
def test_collect_node_metadata(root, expected_result) -> None:
    res = collect_node_metadata(root)
    assert res.is_ok()

    nodes = res.unwrap()
    assert list(map(_convert_node_to_comparable_obj, nodes)) == expected_result
