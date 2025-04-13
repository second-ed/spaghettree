import random
from copy import deepcopy
from typing import Callable

import numpy as np
import pandas as pd
from returns.result import Failure, Success

from spaghettree.metrics import directed_weighted_modularity_df
from spaghettree.processing import clean_calls_df, get_adj_matrix


# @profile_func("time")
def hill_climber_search(
    search_df: pd.DataFrame,
    module_names: tuple,
    class_names: tuple,
    func_names: tuple,
    sims: int = 5000,
) -> tuple[pd.DataFrame, list[dict]]:
    search_df = search_df.copy()

    best_score = get_modularity_score(search_df)
    cand_score = -1

    print(f"    base score: {best_score:.3f}")

    epochs = []

    for _ in range(sims):
        cand_df, cand_score = mutate(search_df, module_names, func_names, class_names)

        if cand_score > best_score:
            best_score = cand_score
            search_df = cand_df.copy()

        epochs.append(
            {
                "cand_score": cand_score,
                "best_score": best_score,
            }
        )

    print(f"    {best_score = }")
    return search_df, pd.DataFrame(epochs), best_score


# @profile_func()
def genetic_search(
    search_df: pd.DataFrame,
    module_names: tuple,
    class_names: tuple,
    func_names: tuple,
    population: int = 50,
    sims: int = 100,
) -> tuple[pd.DataFrame, list[dict]]:
    best_score = get_modularity_score(search_df)
    print(f"    base score: {best_score:.3f}")

    init_pop = [(search_df.copy(), -1) for _ in range(population)]
    curr_gen = init_pop

    epochs = []

    for _ in range(sims):
        mutated = sorted(
            [mutate(df, module_names, func_names, class_names) for df, _ in curr_gen],
            key=lambda x: -x[1],
        )
        best = mutated[: population // 2]
        scores = [b[1] for b in best]
        mean_score = np.mean(scores)

        cand_df, cand_score = best[0]
        if cand_score > best_score:
            best_score = cand_score
            search_df = cand_df.copy()

        doubled_best = best + best
        curr_gen = deepcopy(doubled_best)
        epochs.append({"mean_score": mean_score, "best_score": scores[0]})

    print(f"    {best_score = }")
    return search_df, pd.DataFrame(epochs), best_score


def mutate(
    search_df: pd.DataFrame,
    module_names: tuple[str],
    func_names: tuple[str],
    class_names: tuple[str],
) -> tuple[pd.DataFrame, float]:
    match (bool(func_names), bool(class_names), random.choice((True, False))):
        case (True, _, True):
            cand_df = update_func_loc(
                search_df, random.choice(func_names), random.choice(module_names)
            )
            cand_score = get_modularity_score(cand_df)
            return cand_df, cand_score

        case (_, True, False):
            cand_df = update_class_loc(
                search_df, random.choice(class_names), random.choice(module_names)
            )
            cand_score = get_modularity_score(cand_df)
            return cand_df, cand_score

        case _:
            return search_df, -1


def _update_obj_loc(
    call_df: pd.DataFrame, obj_col: str, obj: str, new_module: str
) -> pd.DataFrame:
    call_df = call_df.copy()
    mask = call_df[obj_col].to_numpy() == obj
    call_df.loc[mask, "module"] = new_module
    return call_df


def update_func_loc(call_df: pd.DataFrame, func: str, new_module: str) -> pd.DataFrame:
    return _update_obj_loc(call_df, "func_method", func, new_module)


def update_class_loc(
    call_df: pd.DataFrame, class_: str, new_module: str
) -> pd.DataFrame:
    return _update_obj_loc(call_df, "class", class_, new_module)


def get_modularity_score(
    calls_df: pd.DataFrame, fitness_func: Callable = directed_weighted_modularity_df
) -> float:
    res = Success(calls_df).bind(clean_calls_df).bind(get_adj_matrix).bind(fitness_func)

    match res:
        case Success(score):
            return score
        case Failure(_):
            return -1
        case _:
            raise RuntimeError(f"Invalid return item: {res}")
