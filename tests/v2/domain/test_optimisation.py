import polars as pl
import pytest
from hypothesis import given

from src.spaghettree.v2.domain.optimisation import DirectedWeightedModularity
from tests.conftest import st_dwm_and_comms


@pytest.mark.parametrize(
    "communities, expected_result",  # noqa: PT006
    [
        pytest.param(
            pl.DataFrame(
                [
                    {"node": 0, "module": 0},
                    {"node": 1, "module": 1},
                    {"node": 2, "module": 1},
                    {"node": 3, "module": 1},
                    {"node": 4, "module": 1},
                    {"node": 5, "module": 2},
                    {"node": 6, "module": 2},
                ]
            ),
            0.25,
        ),
        pytest.param(
            pl.DataFrame(
                [
                    {"node": 0, "module": 0},
                    {"node": 1, "module": 1},
                    {"node": 2, "module": 2},
                    {"node": 3, "module": 1},
                    {"node": 4, "module": 1},
                    {"node": 5, "module": 2},
                    {"node": 6, "module": 2},
                ]
            ),
            0.0,
        ),
    ],
)
def test_directed_weighted_modularity(communities, expected_result):
    edges = pl.DataFrame(
        [
            {"src": 1, "dst": 2, "weight": 1},
            {"src": 5, "dst": 6, "weight": 2},
            {"src": 3, "dst": 6, "weight": 1},
        ]
    )

    dwm = DirectedWeightedModularity.from_edges(edges)
    assert dwm.calc(communities) == expected_result


@given(args=st_dwm_and_comms())
def test_get_dwm_is_within_bounds(args) -> None:
    dwm, comms = args
    val = dwm.calc(comms)
    assert -0.5 <= val <= 1.0
