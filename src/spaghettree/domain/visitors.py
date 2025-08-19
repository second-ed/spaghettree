from __future__ import annotations

import attrs
import libcst as cst


@attrs.define
class CallVisitor(cst.CSTVisitor):
    calls: list[str] = attrs.field(factory=list)

    def visit_Call(self, node: cst.Call) -> None:  # noqa: N802
        full_name = self._resolve_attr(node.func)
        if full_name:
            self.calls.append(full_name)

    def _resolve_attr(self, node: cst.BaseExpression) -> str | None:
        if isinstance(node, cst.Name):
            return node.value
        if isinstance(node, cst.Attribute):
            parent = self._resolve_attr(node.value)
            return f"{parent}.{node.attr.value}" if parent else node.attr.value
        return None
