from __future__ import annotations

from copy import deepcopy

import libcst as cst
import numpy as np
from tqdm import tqdm

from spaghettree import safe
from spaghettree.domain.adj_mat import AdjMat
from spaghettree.domain.entities import ClassCST, FuncCST, GlobalCST, ModuleCST
from spaghettree.domain.visitors import CallVisitor

EntityCST = FuncCST | ClassCST | GlobalCST


def str_to_cst(code: str) -> cst.Module:
    return cst.parse_module(code)


def cst_to_str(node: cst.CSTNode) -> str:
    return cst.Module([]).code_for_node(node)


@safe
def create_module_cst_objs(src_code: dict[str, str]) -> dict[str, ModuleCST]:
    def get_module_name(path: str) -> str:
        return path.split("src")[-1].replace("/", ".").removesuffix(".py").strip(".")

    def get_func_cst(parent_name: str, tree: cst.FunctionDef) -> FuncCST:
        cv = CallVisitor()
        tree.visit(cv)
        return FuncCST(f"{parent_name}.{tree.name.value}", tree, cv.calls)

    modules: dict[str, ModuleCST] = {}

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
            if resolved_call := import_map.get(call.split(".")[-1]):
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

    modules = deepcopy(modules)
    modified_modules = {}

    for name, mod in tqdm(modules.items(), "resolving calls"):
        mod = deepcopy(mod)

        import_map = {
            i.as_name: f"{i.module}.{i.as_name}" if i.module != i.as_name else i.module
            for i in mod.imports
        }
        func_map = {fn.name.split(".")[-1]: fn.name for fn in mod.funcs}
        cls_map = {cls.name.split(".")[-1]: cls.name for cls in mod.classes}
        ent_map = {**func_map, **cls_map}

        for fn in mod.funcs:
            fn.calls = resolve_calls(fn.calls, import_map, ent_map)

        for class_ in mod.classes:
            for fn in class_.methods:
                fn.calls = resolve_calls(fn.calls, import_map, ent_map)

        modified_modules[name] = mod
    return modified_modules


@safe
def extract_entities(modules: dict[str, ModuleCST]) -> dict[str, EntityCST]:
    modules = deepcopy(modules)
    entities: dict[str, EntityCST] = {}

    for mod in modules.values():
        for fn in mod.funcs:
            fn.imports = mod.imports
            entities[fn.name] = fn

        for class_ in mod.classes:
            class_.imports = mod.imports
            entities[class_.name] = class_

        for gbl in mod.global_vars:
            entities[gbl.name] = gbl

    return entities


@safe
def filter_non_native_calls(
    entities: dict[str, EntityCST],
) -> dict[str, EntityCST]:
    entities = deepcopy(entities)
    return {
        name: ent.filter_native_calls(entities).resolve_native_imports()
        for name, ent in entities.items()
    }


@safe
def create_call_tree(entities: dict[str, EntityCST]) -> dict[str, list[str]]:
    return {name: ent.get_call_tree_entries() for name, ent in entities.items()}


@safe
def pair_exclusive_calls(adj_mat: AdjMat) -> AdjMat:
    adj_mat = deepcopy(adj_mat)
    matrix: np.ndarray = adj_mat.mat
    communities: list[int] = adj_mat.communities.copy()

    # make it so we don't weight by call count yet
    adj_bin = (matrix > 0).astype(bool)
    communities: np.ndarray = np.array(communities, dtype=int)

    changed = True
    while changed:
        changed = False

        out_deg = adj_bin.sum(axis=1)
        in_deg = adj_bin.sum(axis=0)

        rows, cols = np.where((out_deg == 1)[:, None] & adj_bin & (in_deg == 1))

        for a, b in zip(rows, cols):
            if communities[b] != communities[a]:
                communities[communities == communities[b]] = communities[a]
                changed = True

    adj_mat.communities = communities.tolist()
    return adj_mat
