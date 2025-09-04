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


@attrs.define(frozen=True, eq=True, order=True)
class EntityLocation:
    path: str = attrs.field()
    name: str = attrs.field(eq=False)
    line_no: int = attrs.field()


class LocationVisitor(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (cst.metadata.PositionProvider,)

    def __init__(self, path: str) -> None:
        self.path = path
        self.results = []
        self.depth = 0
        super().__init__()

    def visit_IndentedBlock(self, _: cst.IndentedBlock) -> bool | None:  # noqa: N802
        self.depth += 1

    def leave_IndentedBlock(self, _: cst.IndentedBlock) -> bool | None:  # noqa: N802
        self.depth -= 1

    def visit_ClassDef(self, node: cst.ClassDef) -> None:  # noqa: N802
        self.results.append(
            EntityLocation(
                path=self.path,
                name=node.name.value,
                line_no=self.get_metadata(cst.metadata.PositionProvider, node).start.line,
            )
        )

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:  # noqa: N802
        self.results.append(
            EntityLocation(
                path=self.path,
                name=node.name.value,
                line_no=self.get_metadata(cst.metadata.PositionProvider, node).start.line,
            )
        )

    def visit_Assign(self, node: cst.Assign) -> None:  # noqa: N802
        for target in node.targets:
            if isinstance(target.target, cst.Name) and self.depth == 0:
                self.results.append(
                    EntityLocation(
                        path=self.path,
                        name=target.target.value,
                        line_no=self.get_metadata(
                            cst.metadata.PositionProvider, target.target
                        ).start.line,
                    )
                )
