from __future__ import annotations

import attrs
import libcst as cst

from spaghettree.domain.entities import (
    ClassCST,
    FuncCST,
    GlobalCST,
    ImportCST,
    ImportType,
    scope_to_str,
)


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
    scope: list[str] = attrs.field(factory=list)
    depth: int = attrs.field(default=0)
    in_cls: bool = attrs.field(default=False)
    in_func: bool = attrs.field(default=False)
    in_global: bool = attrs.field(default=False)
    entities: dict = attrs.field(factory=dict)
    locations: dict = attrs.field(factory=dict)
    imports: set[ImportCST] = attrs.field(factory=set)

    def __attrs_post_init__(self) -> None:
        self.scope.append(self.module_name)

    @property
    def in_method(self) -> bool:
        return self.in_cls and self.in_func

    @property
    def is_toplevel(self) -> bool:
        return self.depth == 0

    def visit_IndentedBlock(self, _: cst.IndentedBlock) -> bool | None:  # noqa: N802
        self.depth += 1

    def leave_IndentedBlock(self, _: cst.IndentedBlock) -> bool | None:  # noqa: N802
        self.depth -= 1

    def visit_Import(self, node: cst.Import) -> None:  # noqa: N802
        self._record_import(None, node.names, ImportType.IMPORT)

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:  # noqa: N802
        module = self._resolve_attr(node.module)
        if module is None:
            # skip relative imports
            return

        if isinstance(node.names, cst.ImportStar):
            self._add_import(module, ImportType.FROM, "*", "*")
            return

        aliases = [node.names] if isinstance(node.names, cst.ImportAlias) else node.names
        self._record_import(module, aliases, ImportType.FROM)

    def _record_import(self, module: str | None, aliases: list, import_type: ImportType) -> None:
        for alias in aliases:
            name = self._resolve_attr(alias.name)
            asname = alias.asname.name.value if alias.asname else name
            self._add_import(module or name, import_type, name, asname)

    def _add_import(self, key: str, import_type: ImportType, name: str, as_name: str) -> None:
        self.imports.add(ImportCST(key, import_type, name, as_name))

    def visit_Assign(self, node: cst.Assign) -> None:  # noqa: N802
        if not self.is_toplevel:
            return
        for target in node.targets:
            if not isinstance(target.target, cst.Name) or target.target.value == "__all__":
                continue
            self._record_global(target.target.value, node)

    def leave_Assign(self, _: cst.Assign) -> None:  # noqa: N802
        self._leave_global()

    def visit_AnnAssign(self, node: cst.AnnAssign) -> None:  # noqa: N802
        if self.is_toplevel and isinstance(node.target, cst.Name):
            self._record_global(node.target.value, node)

    def leave_AnnAssign(self, _: cst.AnnAssign) -> None:  # noqa: N802
        self._leave_global()

    def _record_global(self, name: str, node: cst.CSTNode) -> None:
        self.in_global = True
        self.scope.append(name)
        scope = self._get_current_scope()
        self.entities[scope] = GlobalCST(scope, node)
        self._record_location(node, name)

    def _leave_global(self) -> None:
        if self.in_global and len(self.scope) > 1:
            self.scope.pop()
            self.in_global = False

    def visit_ClassDef(self, node: cst.ClassDef) -> None:  # noqa: N802
        if self.is_toplevel:
            self.scope.append(node.name.value)
            self.in_cls = True
            scope = self._get_current_scope()
            bases = [self._resolve_attr(arg.value) for arg in node.bases if node.bases]
            self.entities[scope] = ClassCST(scope, node, bases=bases)
            self._record_location(node, node.name.value)

    def leave_ClassDef(self, _: cst.ClassDef) -> None:  # noqa: N802
        if self.is_toplevel:
            self.scope.pop()
            self.in_cls = False

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:  # noqa: N802
        if self.is_toplevel or self.in_cls:
            self.scope.append(node.name.value)
            self.in_func = True

        scope = self._get_current_scope()
        func_cst = FuncCST(scope, node)

        if self.in_cls:
            self.entities[self._get_current_class_scope()].methods.append(func_cst)
        elif self.is_toplevel:
            self._record_location(node, node.name.value)
            self.entities[scope] = func_cst

    def leave_FunctionDef(self, _: cst.FunctionDef) -> None:  # noqa: N802
        if self.is_toplevel or self.in_cls:
            self.scope.pop()
            self.in_func = False

    def visit_Call(self, node: cst.Call) -> None:  # noqa: N802
        scope = self._get_current_scope()
        fn_call = self._resolve_attr(node.func)

        if self.in_method:
            # add the calls to the last added method
            self.entities[self._get_current_class_scope()].methods[-1].calls.append(fn_call)
        elif self.in_func:
            self.entities[scope].calls.append(fn_call)

    def visit_Name(self, node: cst.Name) -> None:  # noqa: N802
        scope = self._get_current_class_scope() if self.in_method else self._get_current_scope()

        # class attributes/attrs/dataclasses etc before the first method
        if self.in_cls or self.in_func:
            for imp in self.imports:
                if imp.as_name == node.value:
                    self.entities[scope].imports.add(imp)

        # methods/funcs exist
        if self.in_func:
            if self.in_cls:
                cls_scope = self._get_current_class_scope()

                if node.value not in self.entities[cls_scope].methods[-1].calls:
                    self.entities[cls_scope].methods[-1].calls.append(node.value)

            elif node.value not in self.entities[scope].calls:
                self.entities[scope].calls.append(node.value)

        if self.in_global:
            self.entities[scope].referenced.append(node.value)

    def _get_current_scope(self) -> str:
        return scope_to_str(self.scope)

    def _get_current_class_scope(self) -> str:
        return scope_to_str(self.scope[:-1])

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
        if isinstance(node, cst.Subscript):
            return self._resolve_attr(node.value)
        return None
