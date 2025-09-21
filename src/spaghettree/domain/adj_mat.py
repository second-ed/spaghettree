from typing import Self

import attrs
import numpy as np

from spaghettree import safe
from spaghettree.logger import logger


@attrs.define
class AdjMat:
    mat: np.ndarray = attrs.field()
    node_map: dict[int, str] = attrs.field()
    communities: list[int] = attrs.field()
    comm_map: dict[int, str] = attrs.field(factory=dict)

    @classmethod
    @safe
    def from_call_tree(cls, call_tree: dict[str, list[str]]) -> Self:
        logger.debug(f"{call_tree = }")
        ent_idx: dict[str, int] = {node: i for i, node in enumerate(call_tree)}
        node_map: dict[int, str] = {idx: ent_name for ent_name, idx in ent_idx.items()}

        adj_mat = AdjMat._create_adj_map(call_tree, ent_idx)

        return cls(adj_mat, node_map, list(node_map.keys()))

    @classmethod
    @safe
    def from_call_tree_no_optimisation(cls, call_tree: dict[str, list[str]]) -> Self:
        logger.debug(f"{call_tree = }")
        ent_idx: dict[str, int] = {node: i for i, node in enumerate(call_tree)}
        node_map: dict[int, str] = {idx: ent_name for ent_name, idx in ent_idx.items()}

        modules = [".".join(k.split(".")[:-1]) for k in call_tree]
        unique_mods = list(dict.fromkeys(modules))
        mod_map = {name: idx for idx, name in enumerate(unique_mods)}
        communities = [mod_map[name] for name in modules]

        adj_mat = AdjMat._create_adj_map(call_tree, ent_idx)

        return cls(adj_mat, node_map, communities, comm_map={v: k for k, v in mod_map.items()})

    @staticmethod
    def _create_adj_map(call_tree: dict[str, list[str]], ent_idx: dict[str, int]) -> np.ndarray:
        n = len(ent_idx)
        adj_mat = np.zeros((n, n), dtype=int)

        for caller, called in call_tree.items():
            for call in called:
                src_idx = ent_idx[caller]
                dst_idx = ent_idx[call]
                adj_mat[src_idx, dst_idx] += 1
        return adj_mat
