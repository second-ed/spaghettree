import random

import numpy as np
import pandas as pd
from returns.result import Failure, Success

from spaghettree.__main__ import clean_calls_df, get_adj_matrix
from spaghettree.metrics import directed_weighted_modularity_df


def update_func_loc(call_df: pd.DataFrame, func: str, new_module: str) -> pd.DataFrame:
    call_df = call_df.copy()
    call_df.loc[call_df["func_method"] == func, "module"] = new_module
    return call_df


def update_class_loc(
    call_df: pd.DataFrame, class_: str, new_module: str
) -> pd.DataFrame:
    call_df = call_df.copy()
    call_df.loc[call_df["class"] == class_, "module"] = new_module
    return call_df


def get_modularity_score(calls_df: pd.DataFrame) -> float:
    res = (
        Success(calls_df)
        .bind(clean_calls_df)
        .bind(get_adj_matrix)
        .bind(directed_weighted_modularity_df)
    )

    match res:
        case Success(score):
            return score
        case Failure(_):
            return -1
        case _:
            raise RuntimeError("Invalid return item")


def optimise_graph(
    search_df: pd.DataFrame,
    module_names: tuple,
    class_names: tuple,
    func_names: tuple,
    min_temp: float = 0.01,
    max_temp: float = 1,
    alpha: float = 0.999,
) -> pd.DataFrame:
    search_df = search_df.copy()

    temp = max_temp
    best_score = get_modularity_score(search_df)
    cand_score = -np.inf

    print(f"base score: {best_score:.3f}")

    epochs = []

    while temp > min_temp:
        if random.choice((True, False)):
            cand_df = update_func_loc(
                search_df, random.choice(func_names), random.choice(module_names)
            )
            cand_score = get_modularity_score(cand_df)
        else:
            cand_df = update_class_loc(
                search_df, random.choice(class_names), random.choice(module_names)
            )
            cand_score = get_modularity_score(cand_df)

        if cand_score > best_score:
            print(f"new high score: {cand_score:.3f}")
            best_score = cand_score
            search_df = cand_df.copy()

        # TODO: figure out if I want to add simulated annealing
        # else:
        #     delta_energy = best_score - cand_score
        #     prob = np.exp(-delta_energy / temp)

        #     if np.random.uniform(0, 1) < prob:
        #         search_df = cand_df.copy()
        #         best_score = cand_score

        temp *= alpha
        epochs.append(
            {
                "cand_score": cand_score,
                "best_score": best_score,
            }
        )

    return search_df, epochs
