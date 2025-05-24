from pathlib import Path

import pytest

from src.spaghettree.__main__ import process_package

REPO_ROOT = Path(__file__).parents[1]
MOCK_PACKAGE_PATH = REPO_ROOT.joinpath("mock_package/src/mock_package")


@pytest.fixture(scope="module")
def process_mock_package():
    return process_package(MOCK_PACKAGE_PATH)


def test_process_package(process_mock_package):
    record, result = process_mock_package
    assert len(record) > 0
    assert len(result) > 0


@pytest.mark.parametrize(
    "key, expected_value",
    (
        pytest.param("package_name", "mock_package"),
        pytest.param("n_modules", 2),
        pytest.param("n_classes", 1),
        pytest.param("n_funcs", 5),
        pytest.param("n_calls", 7),
        pytest.param("n_calls_package_funcs", 5),
        pytest.param("total_sims", 8000),
        pytest.param("initial_population_size", 8),
        pytest.param("generations", 1000),
    ),
)
def test_correct_config(process_mock_package, key, expected_value):
    record, _ = process_mock_package
    assert record[key] == expected_value


@pytest.mark.parametrize(
    "search_method",
    (
        pytest.param(
            "hill_climber_search_dwm",
            id="ensure hill_climber_search_dwm improves over base score",
        ),
        pytest.param(
            "sim_annealing_search_dwm",
            id="ensure sim_annealing_search_dwm improves over base score",
        ),
        pytest.param(
            "genetic_search_dwm",
            id="ensure genetic_search_dwm improves over base score",
        ),
    ),
)
def test_optimisation_improves(process_mock_package, search_method):
    record, _ = process_mock_package
    assert record[search_method] > record["base_dwm"]


@pytest.mark.parametrize(
    "key",
    (
        pytest.param(
            "mock_package_hill_climber",
            id="ensure `mock_package_hill_climber` finds expected communities",
        ),
        pytest.param(
            "mock_package_sim_annealing",
            id="ensure `mock_package_sim_annealing` finds expected communities",
        ),
        pytest.param(
            "mock_package_genetic",
            id="ensure `mock_package_genetic` finds expected communities",
        ),
    ),
)
def test_communities(process_mock_package, key):
    _, result_obj = process_mock_package
    res_comms = dict(result_obj[key].search_df.groupby("module")["func_method"].agg(list))
    assert res_comms == {
        "module_a": ["func_a", "func_b"],
        "module_b": ["func_c", "func_e", "func_d", "method_a", "method_a"],
    }
