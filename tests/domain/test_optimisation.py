import hypothesis.extra.numpy as hnp
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from spaghettree.domain.optimisation import (
    AdjMat,
    SuggestedMerge,
    get_dwm,
    get_merge_pairs,
    get_top_suggested_merges,
    optimise_communities,
)
from tests.conftest import identifier


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
                    gain=0.15999999999999986,
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
        pytest.param(
            {
                "case_9.mod_generics.process_list": ["case_9.mod_utils.process_data"],
                "case_9.mod_generics.create_mapping": [],
                "case_9.mod_generics.find_item": [],
                "case_9.mod_utils.T": [],
                "case_9.mod_utils.process_data": ["case_9.mod_utils.T"],
                "case_9.mod_utils.get_first_item": ["case_9.mod_utils.T"],
            },
            [
                SuggestedMerge(
                    entity="case_9.mod_utils.process_data",
                    target_community="case_9.mod_generics",
                    gain=0.22222222222222215,
                )
            ],
        ),
    ],
)
def test_get_top_suggested_merges(call_tree, expected_result):
    adj_mat = AdjMat.from_call_tree(call_tree, optimise=False).unwrap()
    res = get_top_suggested_merges(adj_mat)
    assert res.is_ok()
    assert res.unwrap() == expected_result


@st.composite
def st_adj_mat_and_comms(draw, max_n: int = 20, max_val: int = 20) -> tuple[np.ndarray, list[int]]:
    n = draw(st.integers(min_value=1, max_value=max_n))

    adj_mat = draw(
        hnp.arrays(
            dtype=np.int64,
            shape=(n, n),
            elements=st.integers(min_value=0, max_value=max_val),
        )
    )

    comms = draw(
        st.lists(
            st.integers(min_value=1, max_value=max_val),
            min_size=n,
            max_size=n,
        )
    )

    return adj_mat, comms


@given(st_adj_mat_and_comms())
def test_get_dwm_is_within_bounds(data):
    mat, comms = data
    dwm = get_dwm(mat, comms)
    assert -0.5 <= dwm <= 1.0


def st_call_tree():
    subs_s = st.lists(identifier, min_size=1, max_size=4, unique=True)
    mods_s = st.lists(identifier, min_size=2, max_size=5, unique=True)
    ents_s = st.lists(identifier, min_size=5, max_size=20, unique=True)

    def _make_paths(subs, mods, ents):
        path_strategy = st.builds(
            lambda s, m, e: f"package.{s}.{m}.{e}",
            st.sampled_from(subs),
            st.sampled_from(mods),
            st.sampled_from(ents),
        )
        return st.sets(path_strategy, min_size=len(ents), max_size=len(ents)).map(sorted)

    return (
        st.tuples(subs_s, mods_s, ents_s)
        .flatmap(lambda x: _make_paths(*x))
        .flatmap(
            lambda k: st.fixed_dictionaries(
                {key: st.lists(st.sampled_from(k), min_size=0, max_size=5) for key in k}
            )
        )
    )


@given(st_call_tree())
def test_does_not_produce_worse_dwm(tree):
    adj_mat = AdjMat.from_call_tree(tree, optimise=True).unwrap()
    starting_dwm = get_dwm(adj_mat.mat, adj_mat.communities)
    res = optimise_communities(adj_mat)

    assert res.is_ok()

    res_adj_mat = res.unwrap()
    final_dwm = get_dwm(res_adj_mat.mat, res_adj_mat.communities)
    assert starting_dwm <= final_dwm


@given(st_call_tree())
def test_possible_merges_improve_dwm(tree):
    adj_mat = AdjMat.from_call_tree(tree, optimise=True).unwrap()
    merge_pairs = get_merge_pairs(adj_mat)

    if merge_pairs:
        assert all(pair.gain > 0 for pair in merge_pairs)


@given(st_call_tree())
def test_suggested_merges_improve_dwm(tree):
    adj_mat = AdjMat.from_call_tree(tree, optimise=False).unwrap()
    suggested_merges = get_top_suggested_merges(adj_mat)

    assert suggested_merges.is_ok()
    suggested_merges = suggested_merges.unwrap()

    if suggested_merges:
        assert all(pair.gain > 0 for pair in suggested_merges)
