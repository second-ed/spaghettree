from typing import Self

import attrs
import numpy as np

from spaghettree import safe


@attrs.define
class AdjMat:
    mat: np.ndarray = attrs.field()
    node_map: dict[int, str] = attrs.field()
    communities: list[int] = attrs.field()

    @classmethod
    @safe
    def from_call_tree(cls, call_tree: dict[str, list[str]]) -> Self:
        ent_idx: dict[str, int] = {node: i for i, node in enumerate(call_tree)}
        node_map: dict[int, str] = {v: k for k, v in ent_idx.items()}
        n = len(ent_idx)
        adj_mat = np.zeros((n, n), dtype=int)

        for caller, called in call_tree.items():
            for call in called:
                src_idx = ent_idx[caller]
                dst_idx = ent_idx[call]
                adj_mat[src_idx, dst_idx] += 1

        return cls(adj_mat, node_map, list(node_map.keys()))
