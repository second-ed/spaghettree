from __future__ import annotations

import os

import libcst as cst
from tqdm import tqdm

from spaghettree.v2 import safe
from spaghettree.v2.domain.cst_parsing.entities import ClassCST, FuncCST, ModuleCST
from spaghettree.v2.domain.cst_parsing.visitors import CallVisitor


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
