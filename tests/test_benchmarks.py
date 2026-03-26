"""Performance benchmarks for spaghettree core operations."""

import types
from pathlib import Path

import numpy as np
import pytest

from spaghettree.adapters.io_wrapper import IOWrapper
from spaghettree.domain.optimisation import (
    AdjMat,
    get_dwm,
    get_merge_pairs,
    get_top_suggested_merges,
    optimise_communities,
)
from spaghettree.domain.parsing import (
    create_call_tree,
    extract_entities_and_locations,
    filter_non_native_calls,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SMALL_CALL_TREE = {
    "case_3.mod_a.A": [],
    "case_3.mod_a.func_a": [],
    "case_3.mod_a.func_b": ["case_3.mod_a.func_a", "case_3.mod_a.func_a"],
    "case_3.mod_b.CONSTANT": [],
    "case_3.mod_b.B": ["case_3.mod_b.CONSTANT"],
    "case_3.mod_b.C": ["case_3.mod_a.A", "case_3.mod_b.B"],
}

MEDIUM_CALL_TREE = {
    "case_2.__init__.__all__": [],
    "case_2.mod_a.func_a": ["case_2.mod_a.func_b"],
    "case_2.mod_a.func_b": [],
    "case_2.mod_a.func_c": ["case_2.mod_b.func_d"],
    "case_2.mod_a.isolated_func": [],
    "case_2.mod_b.func_d": [],
    "case_2.mod_b.ClassA": ["case_2.mod_b.func_d", "case_2.mod_b.func_d"],
    "case_9.mod_generics.process_list": ["case_9.mod_utils.process_data"],
    "case_9.mod_generics.create_mapping": [],
    "case_9.mod_generics.find_item": [],
    "case_9.mod_utils.T": [],
    "case_9.mod_utils.process_data": ["case_9.mod_utils.T"],
    "case_9.mod_utils.get_first_item": ["case_9.mod_utils.T"],
}


@pytest.fixture(scope="session")
def mock_files() -> types.MappingProxyType[str, str]:
    io = IOWrapper()
    files = io.read_files(Path("./mock_data"))
    if files.is_ok():
        return types.MappingProxyType(files.inner)
    raise files.error


# ---------------------------------------------------------------------------
# Benchmarks: core modularity calculation
# ---------------------------------------------------------------------------


def test_bench_get_dwm_small(benchmark):
    """Benchmark get_dwm on a small adjacency matrix."""
    adj_mat = AdjMat.from_call_tree(SMALL_CALL_TREE, optimise=False).unwrap()
    benchmark(get_dwm, adj_mat.mat, adj_mat.communities)


def test_bench_get_dwm_medium(benchmark):
    """Benchmark get_dwm on a medium adjacency matrix."""
    adj_mat = AdjMat.from_call_tree(MEDIUM_CALL_TREE, optimise=False).unwrap()
    benchmark(get_dwm, adj_mat.mat, adj_mat.communities)


def test_bench_get_dwm_large_random(benchmark):
    """Benchmark get_dwm on a large random adjacency matrix."""
    rng = np.random.default_rng(42)
    n = 50
    mat = rng.integers(0, 5, size=(n, n))
    communities = rng.integers(0, 10, size=n).tolist()
    benchmark(get_dwm, mat, communities)


# ---------------------------------------------------------------------------
# Benchmarks: adjacency matrix construction
# ---------------------------------------------------------------------------


def test_bench_adjmat_from_call_tree_small(benchmark):
    """Benchmark AdjMat construction from a small call tree."""
    benchmark(lambda: AdjMat.from_call_tree(SMALL_CALL_TREE, optimise=True).unwrap())


def test_bench_adjmat_from_call_tree_medium(benchmark):
    """Benchmark AdjMat construction from a medium call tree."""
    benchmark(lambda: AdjMat.from_call_tree(MEDIUM_CALL_TREE, optimise=True).unwrap())


# ---------------------------------------------------------------------------
# Benchmarks: merge pair detection
# ---------------------------------------------------------------------------


def test_bench_get_merge_pairs(benchmark):
    """Benchmark finding merge pair candidates."""
    adj_mat = AdjMat.from_call_tree(MEDIUM_CALL_TREE, optimise=True).unwrap()
    benchmark(get_merge_pairs, adj_mat)


def test_bench_get_top_suggested_merges(benchmark):
    """Benchmark computing top suggested merges."""
    adj_mat = AdjMat.from_call_tree(MEDIUM_CALL_TREE, optimise=False).unwrap()
    benchmark(lambda: get_top_suggested_merges(adj_mat).unwrap())


# ---------------------------------------------------------------------------
# Benchmarks: community optimisation
# ---------------------------------------------------------------------------


def test_bench_optimise_communities_small(benchmark):
    """Benchmark full community optimisation on a small call tree."""

    def run():
        adj_mat = AdjMat.from_call_tree(SMALL_CALL_TREE, optimise=True).unwrap()
        return optimise_communities(adj_mat).unwrap()

    benchmark(run)


def test_bench_optimise_communities_medium(benchmark):
    """Benchmark full community optimisation on a medium call tree."""

    def run():
        adj_mat = AdjMat.from_call_tree(MEDIUM_CALL_TREE, optimise=True).unwrap()
        return optimise_communities(adj_mat).unwrap()

    benchmark(run)


# ---------------------------------------------------------------------------
# Benchmarks: parsing pipeline
# ---------------------------------------------------------------------------


def test_bench_extract_entities(benchmark, mock_files):
    """Benchmark entity extraction from mock source files."""
    subset = {k: v for k, v in mock_files.items() if "mock_case_3" in k}
    benchmark(lambda: extract_entities_and_locations(subset).unwrap())


def test_bench_filter_non_native_calls(benchmark, mock_files):
    """Benchmark filtering non-native calls from parsed entities."""
    subset = {k: v for k, v in mock_files.items() if "mock_case_3" in k}
    entities, _ = extract_entities_and_locations(subset).unwrap()

    benchmark(lambda: filter_non_native_calls(entities).unwrap())


def test_bench_create_call_tree(benchmark, mock_files):
    """Benchmark call tree creation from parsed entities."""
    subset = {k: v for k, v in mock_files.items() if "mock_case_3" in k}
    entities, _ = extract_entities_and_locations(subset).unwrap()
    filtered = filter_non_native_calls(entities).unwrap()

    benchmark(lambda: create_call_tree(filtered).unwrap())
