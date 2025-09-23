from functools import partial

import pytest

from spaghettree.domain.adj_mat import AdjMat
from spaghettree.domain.entities import ClassCST, FuncCST, GlobalCST, ImportCST, ImportType
from spaghettree.domain.parsing import str_to_cst
from spaghettree.domain.processing import (
    add_empty_inits_if_needed,
    convert_to_code_str,
    create_new_filepaths,
    create_new_module_map,
    infer_module_names,
    remap_imports,
    rename_overlapping_mod_names,
)
from spaghettree.domain.visitors import EntityLocation

CASE_3_CALL_TREE = {
    "case_3.mod_a.A": [],
    "case_3.mod_a.func_a": [],
    "case_3.mod_a.func_b": ["case_3.mod_a.func_a", "case_3.mod_a.func_a"],
    "case_3.mod_b.CONSTANT": [],
    "case_3.mod_b.B": ["case_3.mod_b.CONSTANT"],
    "case_3.mod_b.C": ["case_3.mod_a.A", "case_3.mod_b.B"],
}
CASE_3_ENTITIES = {
    "case_3.mod_a.A": ClassCST(
        name="case_3.mod_a.A",
        tree=str_to_cst("class A:\n    pass\n").body[0],
        methods=[],
        imports=set(),
    ),
    "case_3.mod_a.func_a": FuncCST(
        name="case_3.mod_a.func_a",
        tree=str_to_cst("def func_a() -> int:\n    return math.ceil(0.5)\n").body[0],
        calls=[],
        imports={
            ImportCST(module="math", import_type=ImportType.IMPORT, name="math", as_name="math")
        },
    ),
    "case_3.mod_a.func_b": FuncCST(
        name="case_3.mod_a.func_b",
        tree=str_to_cst("def func_b() -> int:\n    return func_a() + func_a()\n").body[0],
        calls=["case_3.mod_a.func_a", "case_3.mod_a.func_a"],
        imports={
            ImportCST(
                module="case_3.mod_a", import_type=ImportType.FROM, name="func_a", as_name="func_a"
            )
        },
    ),
    "case_3.mod_b.CONSTANT": GlobalCST(
        name="case_3.mod_b.CONSTANT",
        tree=str_to_cst("CONSTANT = 3_000\n"),
        referenced=[],
        imports=set(),
    ),
    "case_3.mod_b.B": ClassCST(
        name="case_3.mod_b.B",
        tree=str_to_cst("class B:\n    def method_a(self) -> int:\n        return CONSTANT\n").body[
            0
        ],
        methods=[
            FuncCST(
                name="case_3.mod_b.B.method_a",
                tree=str_to_cst("def method_a(self) -> int:\n        return CONSTANT\n").body[0],
                calls=["case_3.mod_b.CONSTANT"],
                imports=set(),
            )
        ],
        imports={
            ImportCST(
                module="case_3.mod_b",
                import_type=ImportType.FROM,
                name="CONSTANT",
                as_name="CONSTANT",
            )
        },
    ),
    "case_3.mod_b.C": GlobalCST(
        name="case_3.mod_b.C",
        tree=str_to_cst("C = A | B\n").body[0],
        referenced=["case_3.mod_a.A", "case_3.mod_b.B"],
        imports={
            ImportCST(module="case_3.mod_a", import_type=ImportType.FROM, name="A", as_name="A"),
            ImportCST(module="case_3.mod_b", import_type=ImportType.FROM, name="B", as_name="B"),
        },
    ),
}
CASE_3_LOC_MAP = {
    "A": EntityLocation(path="case_3.mod_a", name="A", line_no=4),
    "func_a": EntityLocation(path="case_3.mod_a", name="func_a", line_no=8),
    "func_b": EntityLocation(path="case_3.mod_a", name="func_b", line_no=12),
    "CONSTANT": EntityLocation(path="case_3.mod_b", name="CONSTANT", line_no=3),
    "B": EntityLocation(path="case_3.mod_b", name="B", line_no=6),
    "C": EntityLocation(path="case_3.mod_b", name="C", line_no=11),
}
CASE_3_EXPECTED_RESULT = {
    "some/src/root/case_3/__init__.py": "",
    "some/src/root/case_3/mod_a.py": (
        "from case_3.mod_b import B\nclass A:\n    pass\n\nC = A | B\n"
    ),
    "some/src/root/case_3/mod_a_mod_overflow.py": "import math\n"
    "def func_a() -> int:\n"
    "    return math.ceil(0.5)\n\n"
    "def func_b() -> int:\n"
    "    return func_a() + func_a()\n",
    "some/src/root/case_3/mod_b.py": "CONSTANT = 3_000\n\n"
    "class B:\n"
    "    def method_a(self) -> int:\n"
    "        return CONSTANT\n",
}


@pytest.mark.parametrize(
    ("call_tree", "entities", "location_map", "src_root", "expected_result"),
    [
        pytest.param(
            CASE_3_CALL_TREE,
            CASE_3_ENTITIES,
            CASE_3_LOC_MAP,
            "some/src/root/case_3",
            CASE_3_EXPECTED_RESULT,
        )
    ],
)
def test_second_half_of_processing(call_tree, entities, location_map, src_root, expected_result):
    adj_mat = AdjMat.from_call_tree(call_tree).inner
    adj_mat.communities = [0, 2, 2, 4, 4, 0]
    res = (
        create_new_module_map(adj_mat, entities=entities)
        .and_then(infer_module_names)
        .and_then(rename_overlapping_mod_names)
        .and_then(remap_imports)
        .and_then(
            partial(
                convert_to_code_str,
                order_map=location_map,
            ),
        )
        .and_then(partial(create_new_filepaths, new_root=src_root))
        .and_then(add_empty_inits_if_needed)
    )
    assert res.is_ok()
    assert res.inner == expected_result
