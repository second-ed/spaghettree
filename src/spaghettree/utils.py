from __future__ import annotations

import black
import isort
import libcst as cst


def format_code_str(code_snippet: str) -> str:
    return black.format_str(isort.code(code_snippet), mode=black.FileMode())


def str_to_cst(code: str) -> cst.Module:
    return cst.parse_module(code)


def cst_to_str(node: cst.CSTNode) -> str:
    return cst.Module([]).code_for_node(node)
