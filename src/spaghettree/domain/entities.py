from __future__ import annotations

from collections.abc import Collection
from typing import Self

import attrs
import libcst as cst
from attrs.validators import instance_of

from spaghettree.domain.globals import GlobalCST, GlobalVisitor
from spaghettree.domain.imports import ImportCST, ImportType, ImportVisitor


@attrs.define
class ModuleCST:
    name: str = attrs.field(validator=instance_of(str))
    tree: cst.Module = attrs.field(validator=[instance_of(cst.Module)], repr=False)
    func_trees: dict[str, cst.FunctionDef] = attrs.field(default=None, repr=False)
    class_trees: dict[str, cst.ClassDef] = attrs.field(default=None, repr=False)
    funcs: list[FuncCST] = attrs.field(factory=list)
    classes: list[ClassCST] = attrs.field(factory=list)
    global_vars: list[GlobalCST] = attrs.field(factory=list)
    imports: list[ImportCST] = attrs.field(default=None, repr=False)

    def __attrs_post_init__(self) -> None:
        iv = ImportVisitor()
        cst.Module(
            [
                node
                for node in self.tree.children
                if isinstance(node, cst.SimpleStatementLine)
                and isinstance(node.body[0], (cst.ImportFrom, cst.Import))
            ],
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

        self.global_vars = [
            GlobalCST(
                name=f"{self.name}.{target.target.value if isinstance(target.target, cst.Name) else target.target.attr.value}",
                tree=stmt,
            )
            for stmt in self.tree.body
            if isinstance(stmt, cst.SimpleStatementLine)
            for assign in stmt.body
            if isinstance(assign, (cst.Assign, cst.AnnAssign))
            for target in (assign.targets if isinstance(assign, cst.Assign) else [assign])
            if isinstance(target.target if isinstance(assign, cst.Assign) else target, cst.Name)
        ]
        visitor = GlobalVisitor(self.name, self.global_vars)
        self.tree.visit(visitor)
        self.global_vars = [gbl for gbl in self.global_vars if not gbl.name.endswith(".__all__")]


@attrs.define
class ClassCST:
    name: str = attrs.field(validator=[instance_of(str)])
    tree: cst.ClassDef = attrs.field(validator=[instance_of(cst.ClassDef)], repr=False)
    methods: list[FuncCST] = attrs.field(validator=[instance_of(list)])
    imports: list[ImportCST] = attrs.field(default=None, repr=False)

    def get_call_tree_entries(self) -> list[str]:
        return [call for meth in self.methods for call in meth.calls]

    def filter_native_calls(self, entities: Collection[str]) -> Self:
        for meth in self.methods:
            meth.calls = [call for call in meth.calls if call in entities]
        return self

    def resolve_native_imports(self) -> Self:
        for method in self.methods:
            for call in method.calls:
                call_parts = call.split(".")
                mod_name = ".".join(call_parts[:-1])
                call_name = call_parts[-1]
                self.imports.append(ImportCST(mod_name, ImportType.FROM, call_name, call_name))
        return self


@attrs.define
class FuncCST:
    name: str = attrs.field(validator=[instance_of(str)])
    tree: cst.FunctionDef = attrs.field(validator=[instance_of(cst.FunctionDef)], repr=False)
    calls: list[str] = attrs.field(validator=[instance_of(list)])
    imports: list[ImportCST] = attrs.field(default=None, repr=False)

    def get_call_tree_entries(self) -> list[str]:
        return self.calls

    def filter_native_calls(self, entities: Collection[str]) -> Self:
        self.calls = [call for call in self.calls if call in entities]
        return self

    def resolve_native_imports(self) -> Self:
        for call in self.calls:
            call_parts = call.split(".")
            mod_name = ".".join(call_parts[:-1])
            call_name = call_parts[-1]
            self.imports.append(ImportCST(mod_name, ImportType.FROM, call_name, call_name))
        return self
