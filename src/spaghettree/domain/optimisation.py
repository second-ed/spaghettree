from collections import defaultdict

import attrs
import numpy as np

from spaghettree import safe
from spaghettree.domain.adj_mat import AdjMat


@safe
def optimise_communities(adj_mat: AdjMat) -> AdjMat:
    valid_merges = get_merge_pairs(adj_mat)
    print(f"{get_dwm(adj_mat.mat, adj_mat.communities) = }")

    while valid_merges:
        to_merge = remove_overlapping_pairs(valid_merges)
        adj_mat.communities = apply_merges(adj_mat.communities, to_merge)
        valid_merges = get_merge_pairs(adj_mat)
        print(f"{get_dwm(adj_mat.mat, adj_mat.communities) = }")
    return adj_mat


@safe
def merge_single_entity_communities_if_no_gain_penalty(adj_mat: AdjMat) -> AdjMat:
    communities = np.array(adj_mat.communities)
    base_score = get_dwm(adj_mat.mat, communities)

    grouped: defaultdict[int, list[tuple[int, str]]] = defaultdict(list)

    for k, v in adj_mat.node_map.items():
        grouped[communities[k]].append((k, v))

    updated, min_for_dir = {}, {}

    for comm, items in grouped.items():
        if len(items) == 1:
            num, name = items[0]
            dirname = ".".join(name.split(".")[:-1])
            min_for_dir[dirname] = min(num, min_for_dir.get(dirname, num))
            updated[comm] = min_for_dir[dirname]

    merge_pairs = []

    for c2, c1 in updated.items():
        merged_communities = communities.copy()
        merged_communities[merged_communities == c2] = c1
        score = get_dwm(adj_mat.mat, merged_communities)

        gain = score - base_score
        if gain >= 0:
            merge_pairs.append(PossibleMerge(c1, c2, gain))

    adj_mat.communities = apply_merges(adj_mat.communities, merge_pairs)
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
