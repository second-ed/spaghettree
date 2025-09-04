from __future__ import annotations

import itertools
from copy import deepcopy

import attrs
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

        module.funcs = [get_func_cst(module.name, tree) for tree in module.func_trees.values()]

        module.classes = [
            ClassCST(
                name,
                tree,
                [
                    get_func_cst(name, f)
                    for f in tree.body.children
                    if isinstance(f, cst.FunctionDef)
                ],
            )
            for name, tree in module.class_trees.items()
        ]

        modules[module.name] = module
    return modules


@attrs.define(frozen=True, eq=True, order=True)
class EntityLocation:
    path: str = attrs.field()
    name: str = attrs.field(eq=False)
    line_no: int = attrs.field()


@safe
def get_location_map(src_code: dict[str, str]) -> dict[str, EntityLocation]:
    def get_line_nos(path: str, source: str) -> list[EntityLocation]:
        tree = cst.metadata.MetadataWrapper(cst.parse_module(source))
        result = []

        class Visitor(cst.CSTVisitor):
            METADATA_DEPENDENCIES = (cst.metadata.PositionProvider,)

            def __init__(self) -> None:
                super().__init__()
                self.depth = 0

            def visit_IndentedBlock(self, _: cst.IndentedBlock) -> bool | None:  # noqa: N802
                self.depth += 1

            def leave_IndentedBlock(self, _: cst.IndentedBlock) -> bool | None:  # noqa: N802
                self.depth -= 1

            def visit_ClassDef(self, node: cst.ClassDef) -> None:  # noqa: N802
                result.append(
                    EntityLocation(
                        path=path,
                        name=node.name.value,
                        line_no=self.get_metadata(cst.metadata.PositionProvider, node).start.line,
                    )
                )

            def visit_FunctionDef(self, node: cst.FunctionDef) -> None:  # noqa: N802
                result.append(
                    EntityLocation(
                        path=path,
                        name=node.name.value,
                        line_no=self.get_metadata(cst.metadata.PositionProvider, node).start.line,
                    )
                )

            def visit_Assign(self, node: cst.Assign) -> None:  # noqa: N802
                for target in node.targets:
                    if isinstance(target.target, cst.Name) and self.depth == 0:
                        result.append(  # noqa: PERF401
                            EntityLocation(
                                path=path,
                                name=target.target.value,
                                line_no=self.get_metadata(
                                    cst.metadata.PositionProvider, target.target
                                ).start.line,
                            )
                        )

        tree.visit(Visitor())
        return result

    locations = list(
        itertools.chain.from_iterable([get_line_nos(path, code) for path, code in src_code.items()])
    )
    return {ent.name: ent for ent in locations}


@safe
def resolve_module_calls(modules: dict[str, ModuleCST]) -> dict[str, ModuleCST]:
    def resolve_calls(
        calls: list[str],
        import_map: dict[str, str],
        func_map: dict[str, str],
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

    for name, mod_obj in tqdm(modules.items(), "resolving calls"):
        mod = deepcopy(mod_obj)

        import_map = {
            i.as_name: f"{i.module}.{i.as_name}" if i.module != i.as_name else i.module
            for i in mod.imports
        }
        func_map = {fn.name.split(".")[-1]: fn.name for fn in mod.funcs}
        cls_map = {cls_.name.split(".")[-1]: cls_.name for cls_ in mod.classes}
        ent_map = {**func_map, **cls_map}

        for fn in mod.funcs:
            fn.calls = resolve_calls(fn.calls, import_map, ent_map)

        for cls_ in mod.classes:
            for fn in cls_.methods:
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

        for cls_ in mod.classes:
            cls_.imports = mod.imports
            entities[cls_.name] = cls_

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

        for a, b in zip(rows, cols, strict=False):
            if communities[b] != communities[a]:
                communities[communities == communities[b]] = communities[a]
                changed = True

    adj_mat.communities = communities.tolist()
    return adj_mat
