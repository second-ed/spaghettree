from pathlib import Path

import pytest
from returns.pipeline import is_successful

from src.spaghettree.__main__ import process_package

REPO_ROOT = Path(__file__).parents[1]
MOCK_PACKAGE_PATH = REPO_ROOT.joinpath("mock_package/src/mock_package")


@pytest.fixture(scope="module")
def process_mock_package():
    return process_package(MOCK_PACKAGE_PATH)


def test_process_package(process_mock_package):
    assert is_successful(process_mock_package)


@pytest.mark.parametrize(
    "key, expected_value",
    (
        pytest.param("package_name", "mock_package"),
        pytest.param("n_modules", 3),
        pytest.param("n_classes", 1),
        pytest.param("n_funcs", 5),
        pytest.param("n_calls", 6),
        pytest.param("n_calls_package_funcs", 4),
        pytest.param("total_sims", 8000),
        pytest.param("initial_population_size", 8),
        pytest.param("generations", 1000),
    ),
)
def test_correct_config(process_mock_package, key, expected_value):
    record, _ = process_mock_package.unwrap()
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
    record, _ = process_mock_package.unwrap()
    assert record[search_method] > record["base_dwm"]
