from __future__ import annotations

import os
from copy import deepcopy
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

    for path in tqdm(paths):
        some_tree = get_src_code(path).map(str_to_cst)
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
def get_call_table(modules: dict[str, ModuleCST]) -> Result[pd.DataFrame, Exception]:
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
    return pd.DataFrame(rows).reset_index(drop=True)


@safe
def clean_calls_df(calls: pd.DataFrame) -> Result[pd.DataFrame, Exception]:
    calls = calls.copy()
    calls["full_address_func_method"] = (
        calls["module"] + "." + calls["class"] + "." + calls["func_method"]
    ).str.replace("..", ".")

    full_address_mapping = dict(
        zip(calls["func_method"], calls["full_address_func_method"])
    )
    calls["full_address_calls"] = (
        calls["call"].map(full_address_mapping).fillna(calls["call"])
    )

    calls.loc[
        ~calls["full_address_calls"].isin(calls["full_address_func_method"]),
        "full_address_calls",
    ] = ""
    return calls


@safe
def get_adj_matrix(
    data: pd.DataFrame, delim: str = "."
) -> Result[pd.DataFrame, Exception]:
    funcs = data["full_address_func_method"].to_numpy()
    calls = data["full_address_calls"].to_numpy()

    all_nodes = np.unique(np.concatenate((funcs, calls[calls != ""])))
    node_idx = {node: i for i, node in enumerate(all_nodes)}

    adj_mat = np.zeros((len(all_nodes), len(all_nodes)), dtype=int)

    same_mod = np.array(
        [
            f.split(".")[0] == c.split(".")[0] if c else False
            for f, c in zip(funcs, calls)
        ]
    )

    for i in range(len(data)):
        src = funcs[i]
        tgt = calls[i]
        if not tgt:
            continue
        src_idx = node_idx[src]
        tgt_idx = node_idx[tgt]
        adj_mat[src_idx, tgt_idx] += 1 if same_mod[i] else -1

    formatted_nodes = [n.replace(".", delim) for n in all_nodes]
    return pd.DataFrame(adj_mat, index=formatted_nodes, columns=formatted_nodes)


def collect_names(json):
    def _collect_names(node, prefix=""):
        name = node.get("name", "")
        if name:
            if prefix:
                name = ".".join([prefix, name])
            names.append(name)
        for n in node.get("methods", []):
            _collect_names(n, name)

    names = []

    for node in json.values():
        _collect_names(node)

    return names


def filter_tree(json, defined_names):
    json = deepcopy(json)

    def _filter_calls(node):
        if "calls" in node:
            node["calls"] = [c for c in node.get("calls", []) if c in defined_names]

        for m in node.get("methods", []):
            _filter_calls(m)

    for node in json.values():
        _filter_calls(node)

    return json


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
