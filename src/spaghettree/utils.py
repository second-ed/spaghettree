from __future__ import annotations

import libcst as cst


def cst_to_str(node: cst.CSTNode) -> str:
    return cst.Module([]).code_for_node(node)


def print_color(text: str, colour: str = "yellow") -> None:
    colors = {"red": "\033[91m", "yellow": "\033[93m", "reset": "\033[0m"}
    colour_code = colors.get(colour.strip().lower(), colors["reset"])
    print(f"{colour_code}{text}{colors['reset']}")
