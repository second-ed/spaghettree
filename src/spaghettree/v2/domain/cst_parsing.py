from __future__ import annotations

import os

import attrs
import libcst as cst
from attrs.validators import instance_of
from tqdm import tqdm

from spaghettree.v2 import safe


@attrs.define
class ModuleCST:
    name: str = attrs.field(validator=instance_of(str))
    tree: cst.Module = attrs.field(validator=[instance_of(cst.Module)], repr=False)
    imports: list[cst.Import | cst.ImportFrom] = attrs.field(default=None, repr=False)
    func_trees: dict[tuple[str, str], cst.FunctionDef] = attrs.field(default=None, repr=False)
    class_trees: dict[str, cst.ClassDef] = attrs.field(default=None, repr=False)
    funcs: list[FuncCST] = attrs.field(default=attrs.Factory(list))
    classes: list[ClassCST] = attrs.field(default=attrs.Factory(list))

    def __attrs_post_init__(self):
        self.imports = [
            node
            for node in self.tree.children
            if isinstance(node, cst.SimpleStatementLine)
            and isinstance(node.body[0], (cst.ImportFrom, cst.Import))
        ]
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


@attrs.define
class FuncCST:
    name: str = attrs.field(validator=[instance_of(str)])
    tree: cst.FunctionDef = attrs.field(validator=[instance_of(cst.FunctionDef)], repr=False)
    calls: list[str] = attrs.field(validator=[instance_of(list)])


@attrs.define
class CallVisitor(cst.CSTVisitor):
    depth: int = attrs.field(default=0, validator=[instance_of(int)])  # type: ignore
    calls: list[str] = attrs.field(default=attrs.Factory(list))

    def visit_IndentedBlock(self, node: cst.IndentedBlock) -> None:
        self.depth += 1

    def leave_IndentedBlock(self, original_node: cst.IndentedBlock) -> None:
        self.depth -= 1

    def visit_Call(self, node: cst.Call) -> None:
        if isinstance(node.func, cst.Name):
            self.calls.append(node.func.value)
        elif isinstance(node.func, cst.Attribute):
            if isinstance(node.func.value, cst.Name):
                self.calls.append(f"{node.func.value.value}.{node.func.attr.value}")


@safe
def create_module_cst_objs(src_code: dict[str, str]) -> dict[str, ModuleCST]:
    def str_to_cst(code: str) -> cst.Module:
        return cst.parse_module(code)

    def get_module_name(path: str) -> str:
        return os.path.splitext(os.path.basename(path))[0]

    def get_func_cst(parent_name: str, tree: cst.FunctionDef) -> FuncCST:
        cv = CallVisitor()
        tree.visit(cv)
        return FuncCST(f"{parent_name}.{tree.name.value}", tree, cv.calls)

    modules = {}

    for path, data in tqdm(src_code.items(), "creating objects"):
        tree = str_to_cst(data)
        module = ModuleCST(get_module_name(path), tree)

        for name, tree in module.func_trees.items():
            func = get_func_cst(module.name, tree)
            module.funcs.append(func)

        for name, tree in module.class_trees.items():
            methods = []
            for f in tree.body.children:
                if isinstance(f, cst.FunctionDef):
                    func = get_func_cst(name, f)
                    methods.append(func)

            c_obj = ClassCST(name, tree, methods)
            module.classes.append(c_obj)

        modules[module.name] = module
    return modules
