import pytest

import spaghettree.domain.entities as ent


@pytest.mark.parametrize(
    "entity_cst",
    [
        pytest.param(ent.ClassCST),
        pytest.param(ent.FuncCST),
        pytest.param(ent.GlobalCST),
    ],
)
def test_entities_match_protocol(entity_cst):
    assert isinstance(entity_cst, ent.EntityCST)
