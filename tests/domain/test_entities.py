import string

import pytest
from hypothesis import given
from hypothesis import strategies as st

import spaghettree.domain.entities as ent
from tests.conftest import identifier


@pytest.mark.parametrize(
    "entity_cst", [pytest.param(ent.ClassCST), pytest.param(ent.FuncCST), pytest.param(ent.GlobalCST)]
)
def test_entities_match_protocol(entity_cst):
    assert isinstance(entity_cst, ent.EntityCST)


@given(st.one_of(st.lists(st.text(alphabet=string.ascii_lowercase), min_size=2), identifier))
def test_scope_to_str(data):
    res = ent.scope_to_str(data)
    assert isinstance(res, str)
    assert all(elem in res for elem in data)
