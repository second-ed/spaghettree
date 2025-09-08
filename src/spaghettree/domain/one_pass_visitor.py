from __future__ import annotations

import attrs
import libcst as cst

from spaghettree.domain.entities import ClassCST, FuncCST
from spaghettree.domain.globals import GlobalCST
from spaghettree.domain.imports import ImportCST, ImportType


@attrs.define(frozen=True, eq=True, order=True)
class EntityLocation:
    path: str = attrs.field()
    name: str = attrs.field(eq=False)
    line_no: int = attrs.field()


class MetadataBase(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (cst.metadata.PositionProvider,)


@attrs.define
class OnePassVisitor(MetadataBase):
    module_name: str = attrs.field()
    current_class: str = attrs.field(default="")
    current_func: str = attrs.field(default="")
    current_global: str = attrs.field(default="")
    depth: int = attrs.field(default=0)
    entities: dict = attrs.field(factory=dict)
    locations: list = attrs.field(factory=list)
    imports: list[ImportCST] = attrs.field(factory=list)

    def visit_Import(self, node: cst.Import) -> None:  # noqa: N802
        for alias in node.names:
            name = self._resolve_attr(alias.name)
            asname = alias.asname.name.value if alias.asname else name
            self._add_import(name, ImportType.IMPORT, name, asname)

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:  # noqa: N802
        module = self._resolve_attr(node.module)
        if module is None:
            return  # skip relative imports

        if isinstance(node.names, cst.ImportStar):
            self._add_import(module, ImportType.FROM, "*", "*")
            return

        aliases = node.names
        if isinstance(aliases, cst.ImportAlias):
            aliases = [aliases]

        for alias in aliases:
            name = self._resolve_attr(alias.name)
            asname = alias.asname.name.value if alias.asname else name
            self._add_import(module, ImportType.FROM, name, asname)

    def visit_IndentedBlock(self, _: cst.IndentedBlock) -> bool | None:  # noqa: N802
        self.depth += 1

    def leave_IndentedBlock(self, _: cst.IndentedBlock) -> bool | None:  # noqa: N802
        self.depth -= 1

    def visit_Assign(self, node: cst.Assign) -> None:  # noqa: N802
        if self.depth == 0:
            for target in node.targets:
                if isinstance(target.target, cst.Name):
                    self.current_global = target.target.value
                    self.entities[self._get_current_scope()] = GlobalCST(
                        self._get_current_scope(), node
                    )
                    self._record_location(node, self.current_global)

    def leave_Assign(self, _: cst.Assign) -> None:  # noqa: N802
        self.current_global = ""

    def visit_AnnAssign(self, node: cst.AnnAssign) -> None:  # noqa: N802
        if self.depth == 0 and isinstance(node.target, cst.Name):
            self.current_global = node.target.value
            self.entities[self._get_current_scope()] = GlobalCST(self._get_current_scope(), node)
            self._record_location(node, self.current_global)

    def leave_AnnAssign(self, _: cst.AnnAssign) -> None:  # noqa: N802
        self.current_global = ""

    def visit_ClassDef(self, node: cst.ClassDef) -> None:  # noqa: N802
        self.current_class = node.name.value
        current_scope = self._get_current_scope()
        self.entities[current_scope] = ClassCST(current_scope, node)
        self._record_location(node, node.name.value)

    def leave_ClassDef(self, _: cst.ClassDef) -> None:  # noqa: N802
        self.current_class = ""

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:  # noqa: N802
        self.current_func = node.name.value
        current_scope = self._get_current_scope()
        func_cst = FuncCST(current_scope, node)

        if self.current_class:
            self.entities[self._get_current_class_scope()].methods.append(func_cst)
        else:
            self._record_location(node, node.name.value)
            self.entities[current_scope] = func_cst

    def leave_FunctionDef(self, _: cst.FunctionDef) -> None:  # noqa: N802
        self.current_func = ""

    def visit_Call(self, node: cst.Call) -> None:  # noqa: N802
        current_scope = self._get_current_scope()
        if self.current_class:
            if self.current_func:
                # add the calls to the last added method
                self.entities[self._get_current_class_scope()].methods[-1].calls.append(
                    self._resolve_attr(node.func)
                )
        elif self.current_func:
            self.entities[current_scope].calls.append(self._resolve_attr(node.func))

    def _get_current_scope(self) -> str:
        return f"{self.module_name}.{'.'.join(elem for elem in [self.current_class, self.current_func, self.current_global] if elem)}"

    def _get_current_class_scope(self) -> str | None:
        if self.current_class:
            return f"{self.module_name}.{self.current_class}"
        return None

    def _add_import(self, key: str, import_type: ImportType, name: str, as_name: str) -> None:
        self.imports.append(ImportCST(key, import_type, name, as_name))

    def _record_location(self, node: cst.CSTNode, name: str) -> None:
        self.locations.append(
            EntityLocation(
                path=self.module_name,
                name=name,
                line_no=self.get_metadata(cst.metadata.PositionProvider, node).start.line,
            )
        )

    def _resolve_attr(self, node: cst.BaseExpression) -> str | None:
        if isinstance(node, cst.Name):
            return node.value
        if isinstance(node, cst.Attribute):
            parent = self._resolve_attr(node.value)
            return f"{parent}.{node.attr.value}" if parent else node.attr.value
        return None
