from __future__ import annotations

import numpy as np
import pandas as pd


def get_np_arrays(call_df: pd.DataFrame) -> tuple[np.ndarray, ...]:
    return (
        call_df["module"].to_numpy(),
        call_df["class"].to_numpy(),
        call_df["func_method"].to_numpy(),
        call_df["call"].to_numpy(),
    )  # type: ignore


def clean_calls(
    modules: np.ndarray, classes: np.ndarray, funcs: np.ndarray, calls: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    has_class = classes != ""
    full_func_addr = np.where(
        has_class, modules + "." + classes + "." + funcs, modules + "." + funcs
    )

    full_addr_map = dict(zip(funcs, full_func_addr))
    full_call_addr = np.frompyfunc(lambda x: full_addr_map.get(x, ""), 1, 1)(calls)

    mask = full_call_addr != ""
    full_func_addr = full_func_addr[mask]  # noqa: E711
    full_call_addr = full_call_addr[mask]  # noqa: E711

    return full_func_addr, full_call_addr


def get_adj_matrix(
    full_func_addr: np.ndarray, full_call_addr: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    nodes = sorted(set(np.concatenate((full_func_addr, full_call_addr))))
    node_idx = {node: i for i, node in enumerate(nodes)}

    n = len(nodes)
    adj_mat = np.zeros((n, n), dtype=int)

    for i, tgt in enumerate(full_call_addr):
        src = full_func_addr[i]
        src_idx = node_idx[src]
        tgt_idx = node_idx[tgt]
        adj_mat[src_idx, tgt_idx] += 1
    return adj_mat, nodes
