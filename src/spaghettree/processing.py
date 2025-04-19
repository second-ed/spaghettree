from __future__ import annotations

import os
from typing import Optional

import libcst as cst
import numpy as np
import pandas as pd
from returns.maybe import Some
from returns.result import Failure, Result, Success, safe
from tqdm import tqdm

from spaghettree.data_structures import ClassCST, ModuleCST, get_func_cst
from spaghettree.io import get_src_code
from spaghettree.utils import str_to_cst


def get_modules(paths: list[str]) -> Result[dict[str, ModuleCST], Exception]:
    modules, fails = {}, []

    for path in tqdm(paths, "creating objects"):
        some_tree = (get_src_code(path)).map(str_to_cst)
        match some_tree:
            case Some(tree):
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
            case _:
                fails.append(path)
    if fails:
        return Failure(ValueError(f"Failed to get tree for paths: {fails}"))
    return Success(modules)


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


def clean_calls_np(modules, classes, funcs, calls):
    full_func_addr = np.where(
        classes, modules + "." + classes + "." + funcs, modules + "." + funcs
    )
    full_addr_map = dict(zip(funcs, full_func_addr))
    full_call_addr = np.vectorize(lambda x: full_addr_map.get(x))(calls)

    full_func_addr = full_func_addr[full_call_addr != None]  # noqa: E711
    full_call_addr = full_call_addr[full_call_addr != None]  # noqa: E711

    return full_func_addr, full_call_addr


def get_adj_matrix(full_func_addr, full_call_addr):
    nodes = np.unique(np.concatenate((full_func_addr, full_call_addr)))
    node_idx = {node: i for i, node in enumerate(nodes)}

    n = len(nodes)
    adj_mat = np.zeros((n, n), dtype=int)

    for i, tgt in enumerate(full_call_addr):
        src = full_func_addr[i]
        src_idx = node_idx[src]
        tgt_idx = node_idx[tgt]
        adj_mat[src_idx, tgt_idx] += 1
    return adj_mat, nodes


def get_module_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _get_full_module_name(module) -> Optional[str]:
    if isinstance(module, cst.Attribute):
        return (
            _get_full_module_name(module.value)  # type: ignore
            + "."
            + module.attr.value
        )
    elif isinstance(module, cst.Name):
        return module.value
    return None


def get_entity_names(mods: dict[str, ModuleCST]) -> tuple[str, str, str]:
    func_names, class_names = [], []

    for mod_csts in mods.values():
        func_names.extend(list(mod_csts.func_trees.keys()))
        class_names.extend(list(mod_csts.class_trees.keys()))

    return tuple(mods.keys()), tuple(func_names), tuple(class_names)
