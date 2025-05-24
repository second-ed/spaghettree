import random
from copy import deepcopy
from typing import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from spaghettree.optimisation_layer.metrics import directed_weighted_modularity
from spaghettree.optimisation_layer.processing import (
    clean_calls,
    get_adj_matrix,
    get_np_arrays,
)


def hill_climber_search(
    search_df: pd.DataFrame, module_names: tuple, class_names: tuple, func_names: tuple, sims: int = 8000
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    search_df = search_df.copy()

    best_modules, classes, funcs, calls = get_np_arrays(search_df)
    best_score = get_modularity_score(best_modules, classes, funcs, calls)
    cand_score = -1

    print(f"    base score: {best_score:.3f}")

    epochs = []

    for _ in tqdm(range(sims), "climbing..."):
        cand_modules, cand_score = mutate(best_modules, classes, funcs, calls, module_names, func_names, class_names)

        if cand_score > best_score:
            best_score = cand_score
            best_modules = np.copy(cand_modules)

        epochs.append({"cand_score": cand_score, "best_score": best_score})

    print(f"    {best_score = }")

    search_df["module"] = best_modules
    search_df["class"] = classes
    search_df["func_method"] = funcs
    search_df["call"] = calls

    return search_df, pd.DataFrame(epochs), best_score


def simulated_annealing_search(
    search_df: pd.DataFrame,
    module_names: tuple,
    class_names: tuple,
    func_names: tuple,
    sims: int = 1000,
    temp: float = 1.0,
    cooling_rate: float = 0.999,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    search_df = search_df.copy()

    best_modules, classes, funcs, calls = get_np_arrays(search_df)
    best_score = get_modularity_score(best_modules, classes, funcs, calls)
    curr_score = best_score
    curr_modules = np.copy(best_modules)

    print(f"    base score: {best_score:.3f}")
    epochs = []

    for _ in tqdm(range(sims), "annealing..."):
        cand_modules, cand_score = mutate(best_modules, classes, funcs, calls, module_names, func_names, class_names)

        score_diff = cand_score - curr_score

        accept = False
        if score_diff > 0:
            accept = True
        else:
            prob = np.exp(score_diff / temp)
            accept = np.random.rand() < prob

        if accept:
            curr_modules = np.copy(cand_modules)
            curr_score = cand_score

            if curr_score > best_score:
                best_score = curr_score
                best_modules = np.copy(curr_modules)

        temp *= cooling_rate

        epochs.append(
            {
                "curr_score": curr_score,
                "best_score": best_score,
                "temperature": temp,
            }
        )

    print(f"    {best_score = }")

    search_df["module"] = best_modules
    search_df["class"] = classes
    search_df["func_method"] = funcs
    search_df["call"] = calls

    return search_df, pd.DataFrame(epochs), best_score


def genetic_search(
    search_df: pd.DataFrame,
    module_names: tuple,
    class_names: tuple,
    func_names: tuple,
    population: int = 50,
    sims: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    search_df = search_df.copy()
    best_modules, classes, funcs, calls = get_np_arrays(search_df)
    best_score = get_modularity_score(best_modules, classes, funcs, calls)
    print(f"    base score: {best_score:.3f}")

    init_pop = [(np.copy(best_modules), -1) for _ in range(population)]
    curr_gen = init_pop

    epochs = []

    for _ in tqdm(range(sims), "evolving..."):
        mutated = sorted(
            [
                mutate(
                    modules,
                    classes,
                    funcs,
                    calls,
                    module_names,
                    func_names,
                    class_names,
                )
                for modules, _ in curr_gen
            ],
            key=lambda x: -x[1],
        )
        best = mutated[: population // 2]
        scores = [b[1] for b in best]
        mean_score = np.mean(scores)

        cand_modules, cand_score = best[0]
        if cand_score > best_score:
            best_score = cand_score
            best_modules = np.copy(cand_modules)

        doubled_best = best + best
        curr_gen = deepcopy(doubled_best)
        epochs.append({"mean_score": mean_score, "best_score": scores[0]})

    print(f"    {best_score = }")

    search_df["module"] = best_modules
    search_df["class"] = classes
    search_df["func_method"] = funcs
    search_df["call"] = calls

    return search_df, pd.DataFrame(epochs), best_score


def mutate(
    modules: np.ndarray,
    classes: np.ndarray,
    funcs: np.ndarray,
    calls: np.ndarray,
    module_names: tuple[str],
    func_names: tuple[str],
    class_names: tuple[str],
):
    match (bool(func_names), bool(class_names), random.choice((True, False))):
        case (True, _, True):
            modules = update_module(modules, funcs, random.choice(func_names), random.choice(module_names))
            cand_score = get_modularity_score(modules, classes, funcs, calls)
            return modules, cand_score

        case (_, True, False):
            modules = update_module(
                modules,
                classes,
                random.choice(class_names),
                random.choice(module_names),
            )
            cand_score = get_modularity_score(modules, classes, funcs, calls)
            return modules, cand_score

        case _:
            return modules, -1


def get_modularity_score(
    modules: np.ndarray,
    classes: np.ndarray,
    funcs: np.ndarray,
    calls: np.ndarray,
    fitness_func: Callable = directed_weighted_modularity,
) -> float:
    full_func_addr, full_call_addr = clean_calls(modules=modules, classes=classes, funcs=funcs, calls=calls)
    adj_mat, nodes = get_adj_matrix(full_func_addr, full_call_addr)
    return fitness_func(adj_mat, nodes)


def update_module(modules: np.ndarray, entities: np.ndarray, ent_name: str, new_module: str) -> np.ndarray:
    modules = np.copy(modules)
    modules[entities == ent_name] = new_module
    return modules


def create_random_replicates(
    search_df: pd.DataFrame,
    sims: int = 1000,
    replace: bool = False,
    unique: bool = False,
) -> list[float]:
    modules, classes, funcs, calls = get_np_arrays(search_df.copy())
    size = len(modules)
    if unique:
        modules = np.unique(modules)
    return [
        get_modularity_score(
            np.random.choice(modules, size=size, replace=replace),
            classes,
            funcs,
            calls,
        )
        for _ in range(sims)
    ]
