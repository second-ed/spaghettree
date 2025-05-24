from __future__ import annotations

import numpy as np


def modularity(adj_mat, nodes, delim: str = ".") -> float:
    adj_mat = np.abs(adj_mat)
    degree = adj_mat.sum(axis=0)
    total_edges = degree.sum()

    communities = np.array([col.split(delim)[0] for col in nodes])
    community_mat = communities[:, None] == communities[None, :]

    expected_matrix = np.outer(degree, degree) / total_edges
    modularity_matrix = (adj_mat - expected_matrix) * community_mat
    return modularity_matrix.sum() / total_edges


def directed_weighted_modularity(adj_mat, nodes, delim: str = ".") -> float:
    adj_mat = np.abs(adj_mat)
    out_degree = adj_mat.sum(axis=0)
    in_degree = adj_mat.sum(axis=1)
    total_edges = out_degree.sum()

    communities = np.array([col.split(delim)[0] for col in nodes])
    community_mat = communities[:, None] == communities[None, :]

    expected_matrix = np.outer(out_degree, in_degree) / total_edges
    modularity_matrix = (adj_mat - expected_matrix) * community_mat
    return modularity_matrix.sum() / total_edges
