from __future__ import annotations

import os
import pathlib
from copy import deepcopy

import libcst as cst
import numpy as np
from tqdm import tqdm

from spaghettree import safe
from spaghettree.domain.adj_mat import AdjMat
from spaghettree.domain.entities import EntityCST
from spaghettree.domain.visitors import EntityLocation, OnePassVisitor
from spaghettree.logger import logger


def str_to_cst(code: str) -> cst.Module:
    return cst.parse_module(code)


def cst_to_str(node: cst.CSTNode) -> str:
    return cst.Module([]).code_for_node(node)


@safe
def extract_entities_and_locations(
    src_code: dict[str, str],
) -> tuple[dict[str, EntityCST], dict[str, EntityLocation]]:
    def find_common_prefix(paths: list[str]) -> str:
        logger.debug(f"{paths = }")
        return str(pathlib.Path(os.path.commonpath(paths)).parent)

    def get_module_name(path: str, root: str) -> str:
        return os.path.splitext(path.removeprefix(root))[0].replace("/", ".").strip(".")

    common_prefix = find_common_prefix(src_code.keys())
    logger.debug(f"{common_prefix = }")

    entities: dict[str, EntityCST] = {}
    locations: dict[str, EntityLocation] = {}

    for path, data in tqdm(src_code.items(), "creating objects"):
        tree = cst.metadata.MetadataWrapper(str_to_cst(data))
        module_name = get_module_name(path, common_prefix)
        visitor = OnePassVisitor(module_name)
        tree.visit(visitor)

        entities.update(visitor.entities)
        locations.update(visitor.locations)

        import_map = {
            i.as_name: f"{i.module}.{i.as_name}" if i.module != i.as_name else i.module
            for i in visitor.imports
        }
        ent_map = {ent.name.split(".")[-1]: ent.name for ent in entities.values()}

        logger.debug(f"{import_map = }")
        logger.debug(f"{ent_map = }")

        for ent in visitor.entities.values():
            ent.resolve_calls(import_map, ent_map).add_referenced_imports(visitor.imports)

        logger.debug(f"{visitor.entities = }")
        logger.debug(f"{visitor.imports = }")

    logger.debug(f"{entities = }")
    logger.debug(f"{locations = }")
    return entities, locations


@safe
def filter_non_native_calls(
    entities: dict[str, EntityCST],
) -> dict[str, EntityCST]:
    logger.debug(f"{entities = }")
    entities = deepcopy(entities)
    return {
        name: ent.filter_native_calls(entities).resolve_native_imports()
        for name, ent in entities.items()
    }


@safe
def create_call_tree(entities: dict[str, EntityCST]) -> dict[str, list[str]]:
    logger.debug(f"{entities = }")
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
