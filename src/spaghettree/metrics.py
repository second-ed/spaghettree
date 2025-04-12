from __future__ import annotations

import numpy as np
import pandas as pd
from returns.result import safe


@safe
def modularity_df(adj_df: pd.DataFrame, delim: str = ".") -> float:
    adj_mat = np.abs(adj_df.to_numpy())
    degree = adj_mat.sum(axis=0)
    total_edges = degree.sum()

    col_names = np.array(adj_df.columns)
    communities = np.array([col.split(delim)[0] for col in col_names])
    community_mat = communities[:, None] == communities[None, :]

    expected_matrix = np.outer(degree, degree) / total_edges
    modularity_matrix = (adj_mat - expected_matrix) * community_mat
    return modularity_matrix.sum() / total_edges


@safe
def directed_weighted_modularity_df(adj_df: pd.DataFrame, delim: str = ".") -> float:
    adj_mat = np.abs(adj_df.to_numpy())
    out_degree = adj_mat.sum(axis=0)
    in_degree = adj_mat.sum(axis=1)
    total_edges = out_degree.sum()

    col_names = np.array(adj_df.columns)
    communities = np.array([col.split(delim)[0] for col in col_names])
    community_mat = communities[:, None] == communities[None, :]

    expected_matrix = np.outer(out_degree, in_degree) / total_edges
    diff_matrix = adj_mat - expected_matrix

    modularity_matrix = diff_matrix * community_mat
    return modularity_matrix.sum() / total_edges
