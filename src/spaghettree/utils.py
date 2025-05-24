from __future__ import annotations

import libcst as cst


def cst_to_str(node: cst.CSTNode) -> str:
    return cst.Module([]).code_for_node(node)
