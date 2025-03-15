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

    for module_name, module_data in tqdm(modules.items()):
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
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


@safe
def clean_calls_df(calls: pd.DataFrame) -> Result[pd.DataFrame, Exception]:
    calls["full_address_func_method"] = (
        calls["module"] + "." + calls["class"] + "." + calls["func_method"]
    ).str.replace("..", ".")
    full_address_mapping = dict(
        calls[["func_method", "full_address_func_method"]].to_numpy().tolist()
    )
    calls["full_address_calls"] = calls["call"].apply(
        lambda x: full_address_mapping.get(x, x)
    )
    calls.loc[
        ~calls["full_address_calls"].isin(calls["full_address_func_method"]),
        "full_address_calls",
    ] = ""
    return calls


@safe
def get_adj_matrix(
    data: pd.DataFrame, delim: str = "\n"
) -> Result[pd.DataFrame, Exception]:
    data = data.to_dict("records")
    nodes = sorted(dict.fromkeys(entry["full_address_func_method"] for entry in data))
    node_idx = {node: i for i, node in enumerate(nodes)}

    adj_matrix = np.zeros((len(nodes), len(nodes)), dtype=int)

    for entry in tqdm(data):
        if entry["full_address_calls"]:
            i, j = (
                node_idx[entry["full_address_func_method"]],
                node_idx[entry["full_address_calls"]],
            )
            if (
                entry["full_address_func_method"].split(".")[0]
                == entry["full_address_calls"].split(".")[0]
            ):
                adj_matrix[i, j] = 1
            else:
                adj_matrix[i, j] = -1

    delimited_nodes = [n.replace(".", delim) for n in nodes]
    return pd.DataFrame(adj_matrix, index=delimited_nodes, columns=delimited_nodes)


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
