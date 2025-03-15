from __future__ import annotations

import os
from copy import deepcopy
from typing import Optional

import attrs
import black
import isort
import libcst as cst
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from attrs.validators import instance_of
from tqdm import tqdm


def get_modules(paths: list[str]) -> dict[str, ModuleCST]:
    modules = {}

    for path in tqdm(paths):
        module = ModuleCST(get_module_name(path), str_to_cst(get_src_code(path)))

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


def get_call_table(modules: dict[str, ModuleCST]) -> pd.DataFrame:
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

        for class_ in tqdm(module_data.classes):
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


def get_adj_matrix(data: pd.DataFrame, delim: str = "\n") -> pd.DataFrame:
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


def plot_graph(adj_df: pd.DataFrame, color_map: dict[str, np.array]) -> None:
    plt.figure(figsize=(12, 8))

    G = nx.from_pandas_adjacency(adj_df)
    G.remove_nodes_from(list(nx.isolates(G)))

    pos = nx.forceatlas2_layout(
        G,
        scaling_ratio=2,
        strong_gravity=True,
        dissuade_hubs=True,
    )

    node_colors = [color_map[n.split("\n")[0]] for n in G.nodes]
    nx.draw(
        G,
        pos=pos,
        with_labels=True,
        node_size=4000,
        node_shape="o",
        node_color=node_colors,
        edge_color="gray",
        arrows=True,
        font_size=8,
    )


def format_code_str(code_snippet: str) -> str:
    return black.format_str(isort.code(code_snippet), mode=black.FileMode())


def get_src_code(path: str) -> str:
    try:
        with open(path, "r") as f:
            src_code = f.read()
        return src_code
    except Exception as e:
        print(f"{e} for {path}")
        return None


def save_modified_code(
    modified_code: str, filepath: str, format_code: bool = True
) -> bool:
    try:
        if format_code:
            modified_code = format_code_str(modified_code)
        with open(filepath, "w") as f:
            f.write(modified_code)
        return True
    except Exception as e:
        print(f"{e} for {filepath}")
        return False


def remove_duplicate_calls(calls: list[str]) -> list:
    return list(dict.fromkeys(calls))


def str_to_cst(code: str) -> cst.Module:
    return cst.parse_module(code)


def cst_to_str(node) -> str:
    return cst.Module([]).code_for_node(node)


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
