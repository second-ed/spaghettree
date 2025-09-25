from collections import defaultdict
from copy import deepcopy
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
    def from_call_tree(cls, call_tree: dict[str, list[str]], *, optimise: bool = True) -> Self:
        logger.debug(f"{call_tree = }")
        ent_idx, node_map, modules, mod_map = AdjMat._get_components(call_tree)
        communities = [mod_map[name] for name in modules]

        mat = AdjMat._create_adj_map(call_tree, ent_idx)

        starting_dwm = get_dwm(mat, communities)
        print(cyan("Starting DWM:"), yellow(starting_dwm))  # noqa: T201
        logger.debug(f"{starting_dwm = }")

        communities = list(node_map.keys()) if optimise else communities

        return cls(mat, node_map, communities, comm_map={v: k for k, v in mod_map.items()})

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

    @staticmethod
    def _get_components(
        call_tree: dict[str, list[str]],
    ) -> tuple[dict[str, int], dict[int, str], list[str], dict[str, int]]:
        ent_idx: dict[str, int] = {node: i for i, node in enumerate(call_tree)}
        node_map: dict[int, str] = {idx: ent_name for ent_name, idx in ent_idx.items()}
        modules: list[str] = [".".join(k.split(".")[:-1]) for k in call_tree]
        unique_mods = list(dict.fromkeys(modules))
        return ent_idx, node_map, modules, {name: idx for idx, name in enumerate(unique_mods)}


@safe
def optimise_communities(adj_mat: AdjMat) -> AdjMat:
    valid_merges = get_merge_pairs(adj_mat)

    while valid_merges:
        to_merge = remove_overlapping_pairs(valid_merges)
        adj_mat.communities = apply_merges(adj_mat.communities, to_merge)
        valid_merges = get_merge_pairs(adj_mat)

    opt_dwm = get_dwm(adj_mat.mat, adj_mat.communities)

    print(cyan("Optimised DWM:"), yellow(opt_dwm))  # noqa: T201
    logger.debug(f"{opt_dwm = }")
    logger.debug(f"{adj_mat.communities = }")

    return adj_mat


@safe
def merge_single_entity_communities_if_no_gain_penalty(adj_mat: AdjMat) -> AdjMat:
    communities = np.array(adj_mat.communities)
    base_score = get_dwm(adj_mat.mat, communities)

    grouped: defaultdict[int, list[tuple[int, str]]] = defaultdict(list)

    for idx, ent_name in adj_mat.node_map.items():
        grouped[communities[idx]].append((idx, ent_name))

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

        if gain := (score - base_score) >= 0:
            merge_pairs.append(PossibleMerge(c1, c2, gain))

    adj_mat.communities = apply_merges(adj_mat.communities, merge_pairs)
    return adj_mat


@attrs.define(eq=True, frozen=True)
class PossibleMerge:
    c1: int = attrs.field()
    c2: int = attrs.field()
    gain: float = attrs.field()


def get_merge_pairs(adj_mat: AdjMat) -> list[PossibleMerge]:
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

    logger.debug(f"{merge_scores = }")
    return merge_scores


def remove_overlapping_pairs(pairs: list[PossibleMerge]) -> list[PossibleMerge]:
    pairs = sorted(pairs, key=lambda x: x.gain, reverse=True)

    selected, seen = [], set()

    for pair in pairs:
        if pair.c1 not in seen and pair.c2 not in seen:
            selected.append(pair)
            seen.add(pair.c1)
            seen.add(pair.c2)

    logger.debug(f"{selected = }")
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

    if total_edges == 0:
        return 0

    communities = np.asarray(communities)
    community_mat = np.equal.outer(communities, communities)

    expected_matrix = np.outer(out_degree, in_degree) / total_edges
    modularity_matrix = (mat - expected_matrix)[community_mat]
    return modularity_matrix.sum() / total_edges


@attrs.define(eq=True, frozen=True)
class SuggestedMerge:
    entity: str = attrs.field()
    target_community: str = attrs.field()
    gain: float = attrs.field(converter=float)

    def display(self) -> None:
        print(  # noqa: T201
            cyan(f"{self.entity}"),
            yellow("->"),
            cyan(f"{self.target_community}"),
            green(f"+{self.gain:.3f}"),
        )


@safe
def get_top_suggested_merges(adj_mat: AdjMat, top_n: int = 5) -> list[SuggestedMerge]:
    adj_mat = deepcopy(adj_mat)
    matrix: np.ndarray = adj_mat.mat
    communities: list[int] = adj_mat.communities.copy()
    comm_choices: set[int] = set(communities)

    starting_score = get_dwm(matrix, communities)

    suggested_merges = []

    for idx, comm in enumerate(communities):
        best_choice, best_score = None, starting_score

        for choice in comm_choices:
            if comm == choice:
                continue
            new_comms = communities.copy()
            new_comms[idx] = choice
            score = get_dwm(matrix, new_comms)

            if score > best_score:
                best_choice, best_score = choice, score

        if best_choice is not None:
            suggested_merges.append(
                SuggestedMerge(
                    adj_mat.node_map.get(idx),
                    adj_mat.comm_map.get(best_choice),
                    best_score - starting_score,
                )
            )

    return sorted(suggested_merges, key=lambda x: -x.gain)[:top_n]


def yellow(inp_str: str) -> str:
    return f"\033[33m{inp_str}\033[0m"


def cyan(inp_str: str) -> str:
    return f"\033[36m{inp_str}\033[0m"


def green(inp_str: str) -> str:
    return f"\033[32m{inp_str}\033[0m"
