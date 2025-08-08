from __future__ import annotations

import attrs
import libcst as cst
from attrs.validators import instance_of

from spaghettree.v2.domain.cst_parsing.imports import ImportCST, ImportType, ImportVisitor


@attrs.define
class ModuleCST:
    name: str = attrs.field(validator=instance_of(str))
    tree: cst.Module = attrs.field(validator=[instance_of(cst.Module)], repr=False)
    imports: list[ImportCST] = attrs.field(default=None, repr=False)
    func_trees: dict[tuple[str, str], cst.FunctionDef] = attrs.field(default=None, repr=False)
    class_trees: dict[str, cst.ClassDef] = attrs.field(default=None, repr=False)
    funcs: list[FuncCST] = attrs.field(factory=list)
    classes: list[ClassCST] = attrs.field(factory=list)

    def __attrs_post_init__(self):
        iv = ImportVisitor()
        cst.Module(
            [
                node
                for node in self.tree.children
                if isinstance(node, cst.SimpleStatementLine)
                and isinstance(node.body[0], (cst.ImportFrom, cst.Import))
            ]
        ).visit(iv)
        self.imports = iv.imports

        self.func_trees = {
            f"{self.name}.{node.name.value}": node
            for node in self.tree.children
            if isinstance(node, cst.FunctionDef)
        }
        self.class_trees = {
            f"{self.name}.{node.name.value}": node
            for node in self.tree.children
            if isinstance(node, cst.ClassDef)
        }


@attrs.define
class ClassCST:
    name: str = attrs.field(validator=[instance_of(str)])
    tree: cst.ClassDef = attrs.field(validator=[instance_of(cst.ClassDef)], repr=False)
    methods: list[FuncCST] = attrs.field(validator=[instance_of(list)])
    imports: list[ImportCST] = attrs.field(default=None, repr=False)

    def resolve_native_imports(self) -> None:
        for method in self.methods:
            for call in method.calls:
                call_parts = call.split(".")
                mod_name = ".".join(call_parts[:-1])
                call_name = call_parts[-1]
                self.imports.append(ImportCST(mod_name, ImportType.FROM, call_name, call_name))


@attrs.define
class FuncCST:
    name: str = attrs.field(validator=[instance_of(str)])
    tree: cst.FunctionDef = attrs.field(validator=[instance_of(cst.FunctionDef)], repr=False)
    calls: list[str] = attrs.field(validator=[instance_of(list)])
    imports: list[ImportCST] = attrs.field(default=None, repr=False)

    def resolve_native_imports(self) -> None:
        for call in self.calls:
            call_parts = call.split(".")
            mod_name = ".".join(call_parts[:-1])
            call_name = call_parts[-1]
            self.imports.append(ImportCST(mod_name, ImportType.FROM, call_name, call_name))
