import random
from copy import deepcopy
from functools import partial

import numpy as np
import pandas as pd
from returns.result import Failure, Success
from tqdm import tqdm

from spaghettree.__main__ import clean_calls_df, get_adj_matrix
from spaghettree.metrics import directed_weighted_modularity_df


# @profile_func("tottime")
def hill_climber_search(
    search_df: pd.DataFrame,
    module_names: tuple,
    class_names: tuple,
    func_names: tuple,
    sims: int = 5000,
) -> tuple[pd.DataFrame, list[dict]]:
    search_df = search_df.copy()

    best_score = get_modularity_score(search_df)
    cand_score = -np.inf

    print(f"base score: {best_score:.3f}")

    epochs = []

    for _ in tqdm(range(sims)):
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

    print(f"{best_score = }")
    return search_df, epochs


# @profile_func("tottime")
def genetic_search(
    search_df: pd.DataFrame,
    module_names: tuple,
    class_names: tuple,
    func_names: tuple,
    population: int = 50,
    sims: int = 100,
) -> tuple[pd.DataFrame, list[dict]]:
    best_score = get_modularity_score(search_df)
    print(f"base score: {best_score:.3f}")

    init_mutate = partial(
        mutate,
        module_names=module_names,
        func_names=func_names,
        class_names=class_names,
    )

    init_pop = [(search_df.copy(), -np.inf) for _ in range(population)]
    curr_gen = init_pop

    epochs = []

    for _ in tqdm(range(sims)):
        mutated = sorted([init_mutate(df) for df, _ in curr_gen], key=lambda x: -x[1])
        best = mutated[: population // 2]
        scores = [b[1] for b in best]
        mean_score = np.mean(scores)
        doubled_best = best + best
        curr_gen = deepcopy(doubled_best)
        epochs.append({"best_score": scores[0], "mean_score": mean_score})

    best_df, best_score = curr_gen[0]
    print(f"{best_score = }")
    return best_df, epochs


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
            return search_df, -np.inf


def _update_obj_loc(
    call_df: pd.DataFrame, obj_col: str, obj: str, new_module: str
) -> pd.DataFrame:
    call_df = call_df.copy()
    call_df.loc[call_df[obj_col] == obj, "module"] = new_module
    return call_df


def update_func_loc(call_df: pd.DataFrame, func: str, new_module: str) -> pd.DataFrame:
    return _update_obj_loc(call_df, "func_method", func, new_module)


def update_class_loc(
    call_df: pd.DataFrame, class_: str, new_module: str
) -> pd.DataFrame:
    return _update_obj_loc(call_df, "class", class_, new_module)


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
