from __future__ import annotations

import libcst as cst


def str_to_cst(code: str) -> cst.Module:
    return cst.parse_module(code)


def cst_to_str(node: cst.CSTNode) -> str:
    return cst.Module([]).code_for_node(node)
