import string
import types
from pathlib import Path

import polars as pl
import pytest
from hypothesis import strategies as st
from hypothesis.strategies import composite

from src.spaghettree.adapters.io_wrapper import IOWrapper
from src.spaghettree.v2.domain.optimisation import DirectedWeightedModularity


@pytest.fixture(scope="session")
def fixture_get_files() -> types.MappingProxyType[str, str]:
    io = IOWrapper()
    files = io.read_files(Path("./mock_data"))

    if files.is_ok():
        return types.MappingProxyType(files.inner)
    raise files.error


@pytest.fixture
def fixture_get_subset_files(
    request: pytest.FixtureRequest, fixture_get_files: types.MappingProxyType[str, str]
) -> tuple[str, dict[str, str]]:
    case_name = request.param
    return case_name, {k: v for k, v in fixture_get_files.items() if case_name in k}


identifier = st.text(alphabet=string.ascii_lowercase, min_size=3, max_size=3)


@composite
def st_weighted_edges(draw: st.DrawFn) -> pl.DataFrame:
    nodes = list(range(draw(st.integers(min_value=2, max_value=10))))
    possible_edges = [(src, dst) for src in nodes for dst in nodes if src != dst]
    edges = draw(st.lists(st.sampled_from(possible_edges), unique=True, min_size=2, max_size=50))
    return pl.DataFrame(
        [{"src": src, "dst": dst, "weight": float(draw(st.integers(min_value=1, max_value=4)))} for src, dst in edges]
    )


def build_dwm_st(weighted_edges: pl.DataFrame) -> DirectedWeightedModularity:
    all_nodes = (
        pl.concat(
            [weighted_edges.select(pl.col("src").alias("node")), weighted_edges.select(pl.col("dst").alias("node"))]
        )
        .unique()
        .sort("node")
    )
    k_out = weighted_edges.group_by("src").agg(pl.sum("weight").alias("k_out")).rename({"src": "node"})
    k_in = weighted_edges.group_by("dst").agg(pl.sum("weight").alias("k_in")).rename({"dst": "node"})

    weighted_nodes = (
        all_nodes.join(k_out, on="node", how="left")
        .join(k_in, on="node", how="left")
        .with_columns(pl.col("k_out").fill_null(0), pl.col("k_in").fill_null(0))
    )

    return DirectedWeightedModularity(
        weighted_nodes=weighted_nodes,
        weighted_edges=weighted_edges,
        total_edges=weighted_edges["weight"].sum(),
    )


@composite
def st_dwm_and_comms(draw: st.DrawFn) -> tuple[DirectedWeightedModularity, pl.DataFrame]:
    dwm = draw(st_weighted_edges().map(build_dwm_st))
    nodes = range(dwm.weighted_nodes.select(pl.col("node").max()).item())
    n = len(nodes)
    modules = draw(st.lists(st.integers(min_value=0, max_value=n), min_size=n, max_size=n))
    return dwm, pl.DataFrame([{"node": n, "module": modules[i]} for i, n in enumerate(nodes)])
