import types
from pathlib import Path

import pytest

from spaghettree.adapters.io_wrapper import IOWrapper


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
    return case_name, {k: v for k, v in fixture_get_files.items() if k.startswith(case_name)}
