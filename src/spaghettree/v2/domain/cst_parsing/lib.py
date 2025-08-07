from __future__ import annotations

import os
from copy import deepcopy

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


@safe
def resolve_module_calls(modules: dict[str, ModuleCST]) -> dict[str, ModuleCST]:
    def resolve_calls(
        calls: list[str], import_map: dict[str, str], func_map: dict[str, str]
    ) -> list[str]:
        resolved_calls: list[str] = []
        for call in calls:
            if resolved_call := import_map.get(call.split(".")[0]):
                if resolved_call.split(".")[-1] != call:
                    common_removed = ".".join(resolved_call.split(".")[:-1])
                    resolved_calls.append(f"{common_removed}.{call}".strip("."))
                else:
                    resolved_calls.append(resolved_call)
            elif resolved_call := func_map.get(call.split(".")[0]):
                resolved_calls.append(resolved_call)
            else:
                resolved_calls.append(call)
        return resolved_calls

    modified_modules = {}

    for name, mod in modules.items():
        mod = deepcopy(mod)

        import_map = {
            i.as_name: f"{i.module}.{i.as_name}" if i.module != i.as_name else i.module
            for i in mod.imports
        }
        func_map = {fn.name.split(".")[-1]: fn.name for fn in mod.funcs}

        for fn in mod.funcs:
            fn.calls = resolve_calls(fn.calls, import_map, func_map)

        for class_ in mod.classes:
            for fn in class_.methods:
                fn.calls = resolve_calls(fn.calls, import_map, func_map)

        modified_modules[name] = mod
    return modified_modules
