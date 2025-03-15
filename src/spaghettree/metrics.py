from __future__ import annotations

import numpy as np
import pandas as pd


def modularity_df(adj_df: pd.DataFrame, delim: str = ".") -> float:
    degree = adj_df.sum(axis=0)
    total_edges = degree.sum()

    communities = np.array([col.split(delim)[0] for col in adj_df.columns])
    community_mat = communities[:, None] == communities

    expected_matrix = np.outer(degree, degree) / total_edges
    diff_matrix = adj_df.values - expected_matrix

    modularity_matrix = diff_matrix * community_mat
    return modularity_matrix.sum() / total_edges


def directed_weighted_modularity_df(adj_df: pd.DataFrame, delim: str = ".") -> float:
    out_degree = adj_df.sum(axis=0)
    in_degree = adj_df.sum(axis=1)
    total_edges = out_degree.sum()

    communities = np.array([col.split(delim)[0] for col in adj_df.columns])
    community_mat = communities[:, None] == communities

    expected_matrix = np.outer(out_degree, in_degree) / total_edges
    diff_matrix = adj_df.values - expected_matrix

    modularity_matrix = diff_matrix * community_mat
    return modularity_matrix.sum() / total_edges
