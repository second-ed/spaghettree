from __future__ import annotations

from typing import Collection, Self

import attrs
import libcst as cst

from spaghettree.v2.domain.cst_parsing.imports import ImportCST


@attrs.define(eq=True)
class GlobalCST:
    name: str = attrs.field()
    tree: cst.SimpleStatementLine = attrs.field(repr=False)
    referenced: list[str] = attrs.field(factory=list)
    imports: list[ImportCST] = attrs.field(factory=list)

    def filter_native_calls(self, entities: Collection[str]) -> Self:
        self.referenced = [ref for ref in self.referenced if ref in entities]
        return self

    def resolve_native_imports(self) -> Self:
        return self


@attrs.define
class GlobalVisitor(cst.CSTVisitor):
    module_name: str
    global_vars: list[GlobalCST]
    module_globals: dict[str, GlobalCST] = attrs.field(default=None)
    current_func: str | None = attrs.field(default=None)

    def __attrs_post_init__(self):
        self.module_globals = {gbl.name.split(".")[-1]: gbl for gbl in self.global_vars}

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        self.current_func = node.name.value

    def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
        self.current_func = None

    def visit_Name(self, node: cst.Name) -> None:
        if self.current_func and node.value in self.module_globals:
            self.module_globals[node.value].referenced.append(
                f"{self.module_name}.{self.current_func}"
            )
