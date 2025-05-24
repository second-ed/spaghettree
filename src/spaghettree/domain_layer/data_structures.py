from __future__ import annotations

import attrs
import libcst as cst
from attrs.validators import instance_of


@attrs.define
class ModuleCST:
    name: str = attrs.field(validator=[])
    tree: cst.Module = attrs.field(validator=[instance_of(cst.Module)], repr=False)
    imports: list = attrs.field(default=None, repr=False)
    func_trees: dict = attrs.field(default=None, repr=False)
    class_trees: dict = attrs.field(default=None, repr=False)
    funcs: list = attrs.field(default=None)
    classes: list = attrs.field(default=None)

    def __attrs_post_init__(self):
        self.imports = [
            node
            for node in self.tree.children
            if isinstance(node, cst.SimpleStatementLine)
            and isinstance(node.body[0], (cst.ImportFrom, cst.Import))
        ]
        self.func_trees = {
            node.name.value: node
            for node in self.tree.children
            if isinstance(node, cst.FunctionDef)
        }
        self.class_trees = {
            node.name.value: node
            for node in self.tree.children
            if isinstance(node, cst.ClassDef)
        }
        self.funcs, self.classes = [], []


@attrs.define
class ClassCST:
    name: str = attrs.field(validator=[instance_of(str)])
    tree: cst.ClassDef = attrs.field(validator=[instance_of(cst.ClassDef)], repr=False)
    methods: list = attrs.field(validator=[instance_of(list)])


@attrs.define
class FuncCST:
    name: str = attrs.field(validator=[instance_of(str)])
    tree: cst.FunctionDef = attrs.field(
        validator=[instance_of(cst.FunctionDef)], repr=False
    )
    calls: list = attrs.field(validator=[instance_of(list)])
    internal: bool = attrs.field(default=False, validator=[instance_of(bool)])


@attrs.define
class CallVisitor(cst.CSTVisitor):
    depth: int = attrs.field(default=0, validator=[instance_of(int)])  # type: ignore
    calls: list = attrs.field(default=None)

    def __attrs_post_init__(self):
        self.calls = []

    def visit_IndentedBlock(self, node: cst.IndentedBlock) -> None:
        self.depth += 1

    def leave_IndentedBlock(self, node: cst.IndentedBlock) -> None:
        self.depth -= 1

    def visit_Call(self, node: cst.Call) -> None:
        if isinstance(node.func, cst.Name):
            self.calls.append(node.func.value)
        elif isinstance(node.func, cst.Attribute):
            if isinstance(node.func.value, cst.Name):
                self.calls.append(f"{node.func.value.value}.{node.func.attr.value}")


def get_func_cst(tree):
    cv = CallVisitor()
    tree.visit(cv)
    return FuncCST(tree.name.value, tree, cv.calls)
