from __future__ import annotations

import attrs
import libcst as cst

from spaghettree.domain.entities import ClassCST, FuncCST, GlobalCST, ImportCST, ImportType


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
    locations: dict = attrs.field(factory=dict)
    imports: set[ImportCST] = attrs.field(factory=set)

    def visit_Import(self, node: cst.Import) -> None:  # noqa: N802
        for alias in node.names:
            name = self._resolve_attr(alias.name)
            asname = alias.asname.name.value if alias.asname else name
            self._add_import(name, ImportType.IMPORT, name, asname)

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:  # noqa: N802
        module = self._resolve_attr(node.module)
        if module is None:
            # skip relative imports
            return

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
        if self.depth != 0:
            return
        for target in node.targets:
            if not isinstance(target.target, cst.Name):
                return
            self.current_global = target.target.value
            scope = self._get_current_scope()
            self.entities[scope] = GlobalCST(scope, node)
            self._record_location(node, self.current_global)

    def leave_Assign(self, _: cst.Assign) -> None:  # noqa: N802
        self.current_global = ""

    def visit_AnnAssign(self, node: cst.AnnAssign) -> None:  # noqa: N802
        if self.depth == 0 and isinstance(node.target, cst.Name):
            self.current_global = node.target.value
            scope = self._get_current_scope()
            self.entities[scope] = GlobalCST(scope, node)
            self._record_location(node, self.current_global)

    def leave_AnnAssign(self, _: cst.AnnAssign) -> None:  # noqa: N802
        self.current_global = ""

    def visit_ClassDef(self, node: cst.ClassDef) -> None:  # noqa: N802
        if self.depth == 0:
            self.current_class = node.name.value
            scope = self._get_current_scope()
            bases = [self._resolve_attr(arg.value) for arg in node.bases if node.bases]
            self.entities[scope] = ClassCST(scope, node, bases=bases)
            self._record_location(node, node.name.value)

    def leave_ClassDef(self, _: cst.ClassDef) -> None:  # noqa: N802
        if self.depth == 0:
            self.current_class = ""

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:  # noqa: N802
        if self.depth == 0 or self.current_class:
            self.current_func = node.name.value

        scope = self._get_current_scope()
        func_cst = FuncCST(scope, node)

        if self.current_class:
            self.entities[self._get_current_class_scope()].methods.append(func_cst)
        elif self.depth == 0:
            self._record_location(node, node.name.value)
            self.entities[scope] = func_cst

    def leave_FunctionDef(self, _: cst.FunctionDef) -> None:  # noqa: N802
        if self.current_class or self.depth == 0:
            self.current_func = ""

    def visit_Call(self, node: cst.Call) -> None:  # noqa: N802
        scope = self._get_current_scope()

        if self.current_class and self.current_func:
            # add the calls to the last added method
            self.entities[self._get_current_class_scope()].methods[-1].calls.append(
                self._resolve_attr(node.func)
            )
        elif self.current_func:
            self.entities[scope].calls.append(self._resolve_attr(node.func))

    def visit_Name(self, node: cst.Name) -> None:  # noqa: N802
        scope = self._get_current_class_scope() if self.current_class else self._get_current_scope()

        # class attributes/attrs/dataclasses etc before the first method
        if self.current_class or self.current_func:
            for imp in self.imports:
                if imp.as_name == node.value:
                    self.entities[scope].imports.add(imp)

        # methods/funcs exist
        if self.current_func:
            if self.current_class:
                cls_scope = self._get_current_class_scope()

                if node.value not in self.entities[cls_scope].methods[-1].calls:
                    self.entities[cls_scope].methods[-1].calls.append(node.value)

            elif node.value not in self.entities[scope].calls:
                self.entities[scope].calls.append(node.value)

        if self.current_global:
            self.entities[scope].referenced.append(node.value)

    def _get_current_scope(self) -> str:
        ent = ".".join(
            elem for elem in [self.current_class, self.current_func, self.current_global] if elem
        )
        return f"{self.module_name}.{ent}"

    def _get_current_class_scope(self) -> str:
        return f"{self.module_name}.{self.current_class}"

    def _add_import(self, key: str, import_type: ImportType, name: str, as_name: str) -> None:
        self.imports.add(ImportCST(key, import_type, name, as_name))

    def _record_location(self, node: cst.CSTNode, name: str) -> None:
        self.locations[name] = EntityLocation(
            path=self.module_name,
            name=name,
            line_no=self.get_metadata(cst.metadata.PositionProvider, node).start.line,
        )

    def _resolve_attr(self, node: cst.BaseExpression) -> str | None:
        if isinstance(node, cst.Name):
            return node.value
        if isinstance(node, cst.Attribute):
            parent = self._resolve_attr(node.value)
            return f"{parent}.{node.attr.value}" if parent else node.attr.value
        return None
