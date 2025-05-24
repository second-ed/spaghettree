from __future__ import annotations

import os

import libcst as cst
import pandas as pd
from returns.result import safe
from tqdm import tqdm

from spaghettree.domain_layer.data_structures import ClassCST, ModuleCST, get_func_cst


def str_to_cst(code: str) -> cst.Module:
    return cst.parse_module(code)


@safe
def get_modules(src_code: dict[str, str]) -> dict[str, ModuleCST]:
    def get_module_name(path: str) -> str:
        return os.path.splitext(os.path.basename(path))[0]

    modules = {}

    for path, data in tqdm(src_code.items(), "creating objects"):
        tree = str_to_cst(data)
        module = ModuleCST(get_module_name(path), tree)

        for name, tree in module.func_trees.items():
            func = get_func_cst(tree)
            module.funcs.append(func)

        for name, tree in module.class_trees.items():
            methods = []
            for f in tree.body.children:
                if isinstance(f, cst.FunctionDef):
                    func = get_func_cst(f)
                    methods.append(func)

            c_obj = ClassCST(name, tree, methods)
            module.classes.append(c_obj)

        modules[module.name] = module
    return modules


@safe
def get_call_table(modules: dict[str, ModuleCST]) -> pd.DataFrame:
    rows = []

    for module_name, module_data in modules.items():
        for func in module_data.funcs:
            if not func.calls:
                rows.append(
                    {
                        "module": module_name,
                        "class": "",
                        "func_method": func.name,
                        "call": "",
                    }
                )
            else:
                for call in func.calls:
                    rows.append(
                        {
                            "module": module_name,
                            "class": "",
                            "func_method": func.name,
                            "call": call,
                        }
                    )

        for class_ in module_data.classes:
            for func in class_.methods:
                if not func.calls:
                    rows.append(
                        {
                            "module": module_name,
                            "class": class_.name,
                            "func_method": func.name,
                            "call": "",
                        }
                    )
                else:
                    for call in func.calls:
                        rows.append(
                            {
                                "module": module_name,
                                "class": class_.name,
                                "func_method": func.name,
                                "call": call.replace(
                                    "self.", f"{module_name}.{class_.name}."
                                ),
                            }
                        )
    return pd.DataFrame(rows)
