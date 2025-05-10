from __future__ import annotations

import datetime as dt
import os
import pprint
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from returns.pipeline import is_successful
from returns.result import Failure, Success, safe

from spaghettree.data_structures import OptResult
from spaghettree.io import read_files, save_results
from spaghettree.metrics import modularity
from spaghettree.processing import (
    get_call_table,
    get_entity_names,
    get_modules,
)
from spaghettree.search import (
    create_random_replicates,
    genetic_search,
    get_modularity_score,
    get_np_arrays,
    hill_climber_search,
    simulated_annealing_search,
)

REPO_ROOT = Path(__file__).parents[2]


def main(parallel: bool):
    packages = (
        REPO_ROOT / ".venv/lib/python3.12/site-packages/_pytest",
        REPO_ROOT / ".venv/lib/python3.12/site-packages/attr",
        REPO_ROOT / ".venv/lib/python3.12/site-packages/beartype",
        REPO_ROOT / ".venv/lib/python3.12/site-packages/black",
        REPO_ROOT / ".venv/lib/python3.12/site-packages/class_inspector",
        REPO_ROOT / ".venv/lib/python3.12/site-packages/diagrams",
        REPO_ROOT / ".venv/lib/python3.12/site-packages/dynaconf",
        REPO_ROOT / ".venv/lib/python3.12/site-packages/faker",
        REPO_ROOT / ".venv/lib/python3.12/site-packages/fastapi",
        REPO_ROOT / ".venv/lib/python3.12/site-packages/icecream",
        REPO_ROOT / ".venv/lib/python3.12/site-packages/locust",
        REPO_ROOT / ".venv/lib/python3.12/site-packages/loguru",
        REPO_ROOT / ".venv/lib/python3.12/site-packages/manimlib",
        REPO_ROOT / ".venv/lib/python3.12/site-packages/more_itertools",
        REPO_ROOT / ".venv/lib/python3.12/site-packages/pipx",
        REPO_ROOT / ".venv/lib/python3.12/site-packages/poetry",
        REPO_ROOT / ".venv/lib/python3.12/site-packages/pre_commit",
        REPO_ROOT / ".venv/lib/python3.12/site-packages/prophet",
        REPO_ROOT / ".venv/lib/python3.12/site-packages/pydantic",
        REPO_ROOT / ".venv/lib/python3.12/site-packages/pyupgrade",
        REPO_ROOT / ".venv/lib/python3.12/site-packages/redis",
        REPO_ROOT / ".venv/lib/python3.12/site-packages/rich",
        REPO_ROOT / ".venv/lib/python3.12/site-packages/schedule",
        REPO_ROOT / ".venv/lib/python3.12/site-packages/sherlock_project",
        REPO_ROOT / ".venv/lib/python3.12/site-packages/sqlmodel",
        REPO_ROOT / ".venv/lib/python3.12/site-packages/textual",
        REPO_ROOT / ".venv/lib/python3.12/site-packages/tox",
        REPO_ROOT / ".venv/lib/python3.12/site-packages/typer",
        REPO_ROOT / ".venv/lib/python3.12/site-packages/urllib3",
    )

    if parallel:
        pool = Pool()
        res = pool.map(process_package, packages)
    else:
        res = [process_package(p) for p in packages]
    res = [r.unwrap() for r in res if isinstance(r, Success)]
    fails = [r.failure() for r in res if isinstance(r, Failure)]

    if fails:
        print(f"{len(fails)} failed packages")
        print(fails)

    now = dt.datetime.now().strftime(format="%y%m%d_%H%M")

    results = {}
    for r in res:
        results.update(r[1])

    save_results(results)

    records = [r[0] for r in res]
    res_df = pd.DataFrame(records)

    for col in res_df.columns:
        if col.endswith("_search_dwm"):
            res_df[f"{col}_gain"] = res_df[col] - res_df["base_dwm"]

        if col.endswith("_search_m"):
            res_df[f"{col}_gain"] = res_df[col] - res_df["base_modularity"]

    res_df.to_csv(f"./results/{now}_results.csv", index=False)


@safe
def process_package(
    p: str,
    total_sims: int = 8000,
    pop: int = 8,
    use_hill_climbing: bool = True,
    use_sim_annealing: bool = True,
    use_genetic_search: bool = True,
):
    sims = total_sims // pop

    config = {
        "hill_climber": {
            "use": use_hill_climbing,
            "func": hill_climber_search,
            "kwargs": {"sims": total_sims},
        },
        "sim_annealing": {
            "use": use_sim_annealing,
            "func": simulated_annealing_search,
            "kwargs": {"sims": total_sims},
        },
        "genetic": {
            "use": use_genetic_search,
            "func": genetic_search,
            "kwargs": {
                "population": pop,
                "sims": sims,
            },
        },
    }

    results = {}

    package_name = os.path.basename(p)
    print("*" * 79)
    print(f"{package_name = }")

    modules = read_files(p).bind(get_modules)
    module_names, func_names, class_names = modules.bind(get_entity_names)
    raw_calls = modules.bind(get_call_table)
    if is_successful(raw_calls):
        raw_calls_df = raw_calls.unwrap()
    else:
        print(raw_calls)
        return {}, {}

    modules, classes, funcs, calls = get_np_arrays(raw_calls_df)

    record = {
        "package_name": package_name,
        "n_modules": len(module_names),
        "n_classes": len(class_names),
        "n_funcs": len(func_names),
        "n_calls": len(calls),
        "n_calls_package_funcs": len(calls[calls != ""]),
        "base_dwm": get_modularity_score(modules, classes, funcs, calls),
        "base_modularity": get_modularity_score(
            modules, classes, funcs, calls, modularity
        ),
        "total_sims": total_sims,
        "initial_population_size": pop,
        "generations": sims,
    }
    pprint.pprint(record, sort_dicts=False)

    replicates = create_random_replicates(raw_calls_df, replace=True)
    permutates = create_random_replicates(raw_calls_df)
    unique_replicates = create_random_replicates(
        raw_calls_df, replace=True, unique=True
    )

    for name, conf in config.items():
        if conf["use"]:
            start_time = time.time()
            search_df, epochs, best_score = conf["func"](
                raw_calls_df,
                module_names=module_names,
                class_names=class_names,
                func_names=func_names,
                **conf["kwargs"],
            )
            end_time = time.time()
            record[f"{name}_duration"] = end_time - start_time

            results[f"{package_name}_{name}"] = OptResult(
                package_name,
                name,
                search_df,
                epochs,
                best_score,
                replicates=replicates,
                permutates=permutates,
                unique_replicates=unique_replicates,
            )
            record[f"{name}_search_dwm"] = best_score

            modules, classes, funcs, calls = get_np_arrays(search_df)
            record[f"{name}_search_m"] = get_modularity_score(
                modules, classes, funcs, calls, modularity
            )
            record[f"{name}_pvalue_replicates"] = np.mean(best_score < replicates)
            record[f"{name}_pvalue_permutates"] = np.mean(best_score < permutates)

    print("*" * 79, end="\n\n")
    return record, results


if __name__ == "__main__":
    main(False)
