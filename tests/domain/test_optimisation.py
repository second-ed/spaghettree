import pytest

from spaghettree.domain.adj_mat import AdjMat
from spaghettree.domain.optimisation import SuggestedMerge, get_top_suggested_merges


@pytest.mark.parametrize(
    ("call_tree", "expected_result"),
    [
        pytest.param(
            {
                "case_1.mod_a.func_a": [],
                "case_1.mod_b.func_b": ["case_1.mod_a.func_a", "case_1.mod_a.func_a"],
            },
            [],
        ),
        pytest.param(
            {
                "case_2.__init__.__all__": [],
                "case_2.mod_a.func_a": ["case_2.mod_a.func_b"],
                "case_2.mod_a.func_b": [],
                "case_2.mod_a.func_c": ["case_2.mod_b.func_d"],
                "case_2.mod_a.isolated_func": [],
                "case_2.mod_b.func_d": [],
                "case_2.mod_b.ClassA": ["case_2.mod_b.func_d", "case_2.mod_b.func_d"],
            },
            [
                SuggestedMerge(
                    entity="case_2.mod_a.func_c", target_community="case_2.mod_b", gain=0.125
                )
            ],
        ),
        pytest.param(
            {
                "case_3.mod_a.A": [],
                "case_3.mod_a.func_a": [],
                "case_3.mod_a.func_b": ["case_3.mod_a.func_a", "case_3.mod_a.func_a"],
                "case_3.mod_b.CONSTANT": [],
                "case_3.mod_b.B": ["case_3.mod_b.CONSTANT"],
                "case_3.mod_b.C": ["case_3.mod_a.A", "case_3.mod_b.B"],
            },
            [
                SuggestedMerge(
                    entity="case_3.mod_a.A",
                    target_community="case_3.mod_b",
                    gain=0.1600000000000001,
                )
            ],
        ),
        pytest.param(
            {
                "case_4.mod_a.CONSTANT": [],
                "case_4.mod_a.func_a": ["case_4.mod_a.CONSTANT"],
                "case_4.mod_a.func_b": ["case_4.mod_a.func_a", "case_4.mod_a.func_a"],
                "case_4.mod_b.CONSTANT": [],
                "case_4.mod_b.B": ["case_4.mod_b.CONSTANT"],
            },
            [],
        ),
        pytest.param({"case_5.mod_a.SomeClass": []}, []),
        pytest.param(
            {
                "case_6.mod_a.PI": [],
                "case_6.mod_a.Circle": ["case_6.mod_a.PI"],
                "case_6.mod_b.calculate_circumference": ["case_6.mod_a.PI"],
            },
            [],
        ),
        pytest.param(
            {
                "case_7.mod_protocol.MockProtocol": [],
                "case_7.mod_protocol.Base": [],
                "case_7.mod_protocol.Child": ["case_7.mod_protocol.Base"],
                "case_7.mod_protocol.FreeClass": [],
            },
            [],
        ),
    ],
)
def test_get_top_suggested_merges(call_tree, expected_result):
    adj_mat = AdjMat.from_call_tree_no_optimisation(call_tree).unwrap()
    res = get_top_suggested_merges(adj_mat)
    assert res.is_ok()
    assert res.unwrap() == expected_result
