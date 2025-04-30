from __future__ import annotations

import glob
import os
import pprint
import time

import numpy as np
from returns.result import safe

from spaghettree.data_structures import OptResult
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
    paths = glob.glob(f"{p}/**/**.py", recursive=True)
    modules = get_modules(paths)
    module_names, func_names, class_names = modules.bind(get_entity_names)
    raw_calls = modules.bind(get_call_table)
    raw_calls_df = raw_calls.unwrap()

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
