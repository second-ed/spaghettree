import attrs
import numpy as np

from spaghettree.v2.domain.cst_parsing.adj_mat import AdjMat


def optimise_communities(adj_mat: AdjMat) -> AdjMat:
    valid_merges = get_merge_pairs(adj_mat)

    while valid_merges:
        to_merge = remove_overlapping_pairs(valid_merges)
        adj_mat.communities = apply_merges(adj_mat.communities, to_merge)
        valid_merges = get_merge_pairs(adj_mat)
    return adj_mat


@attrs.define(eq=True, frozen=True)
class PossibleMerge:
    c1: int = attrs.field()
    c2: int = attrs.field()
    gain: float = attrs.field()


def get_merge_pairs(adj_mat) -> list[PossibleMerge]:
    communities = np.array(adj_mat.communities)
    unique_comms = np.unique(communities)
    base_score = get_dwm(adj_mat.mat, communities)

    merge_scores = []

    for i, c1 in enumerate(unique_comms):
        for c2 in unique_comms[i + 1 :]:
            merged_communities = communities.copy()

            merged_communities[merged_communities == c2] = c1
            score = get_dwm(adj_mat.mat, merged_communities)
            gain = score - base_score
            if gain > 0:
                merge_scores.append(PossibleMerge(c1, c2, gain))
    return merge_scores


def remove_overlapping_pairs(pairs: list[PossibleMerge]) -> list[PossibleMerge]:
    pairs = sorted(pairs, key=lambda x: x.gain, reverse=True)

    selected = []
    seen = set()

    for pair in pairs:
        if pair.c1 not in seen and pair.c2 not in seen:
            selected.append(pair)
            seen.add(pair.c1)
            seen.add(pair.c2)
    return selected


def apply_merges(communities: list[int], pairs: list[PossibleMerge]) -> list[int]:
    communities = np.array(communities)

    for pair in pairs:
        communities[communities == pair.c2] = pair.c1
    return communities.tolist()


def get_dwm(mat: np.ndarray, communities: list[int]) -> float:
    out_degree = mat.sum(axis=0)
    in_degree = mat.sum(axis=1)
    total_edges = out_degree.sum()

    communities = np.array(communities)
    community_mat = communities[:, None] == communities[None, :]

    expected_matrix = np.outer(out_degree, in_degree) / total_edges
    modularity_matrix = (mat - expected_matrix) * community_mat
    return modularity_matrix.sum() / total_edges
