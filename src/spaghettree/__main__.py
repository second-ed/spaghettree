from __future__ import annotations

import glob
import os
import pprint

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


def process_package(
    p: str,
    total_sims: int = 8000,
    pop: int = 8,
    use_hill_climbing: bool = True,
    use_sim_annealing: bool = True,
    use_genetic_search: bool = True,
):
    results = {}

    sims = total_sims // pop
    package_name = os.path.basename(p)
    print("*" * 79)
    print(f"{package_name = }")
    paths = glob.glob(f"{p}/**/**.py", recursive=True)
    modules = get_modules(paths)
    module_names, func_names, class_names = modules.bind(get_entity_names)
    raw_calls = modules.bind(get_call_table)
    raw_calls_df = raw_calls.unwrap()

    modules, classes, funcs, calls = get_np_arrays(raw_calls_df)

    replicates = create_random_replicates(raw_calls_df, replace=True)
    permutates = create_random_replicates(raw_calls_df)

    record = {
        "package_name": package_name,
        "n_modules": len(module_names),
        "n_classes": len(class_names),
        "n_funcs": len(func_names),
        # "n_calls": len(calls_df),
        # "n_calls_package_funcs": len(calls_df[calls_df["full_address_calls"] != ""]),
        "base_dwm": get_modularity_score(modules, classes, funcs, calls),
        "base_modularity": get_modularity_score(
            modules, classes, funcs, calls, modularity
        ),
        "total_sims": total_sims,
        "initial_population_size": pop,
        "generations": sims,
    }
    pprint.pprint(record, sort_dicts=False)

    if use_hill_climbing:
        search_df, epochs, best_score = hill_climber_search(
            raw_calls_df,
            module_names=module_names,
            class_names=class_names,
            func_names=func_names,
            sims=total_sims,
        )

        modules, classes, funcs, calls = get_np_arrays(search_df)
        results[f"{package_name}_hillclimber"] = OptResult(
            "hill_climber",
            search_df,
            epochs,
            best_score,
            replicates=replicates,
            permutates=permutates,
        )
        record["hill_climber_search_dwm"] = best_score
        record["hill_climber_search_m"] = get_modularity_score(
            modules, classes, funcs, calls, modularity
        )

    if use_sim_annealing:
        search_df, epochs, best_score = simulated_annealing_search(
            raw_calls_df,
            module_names=module_names,
            class_names=class_names,
            func_names=func_names,
            sims=total_sims,
        )

        modules, classes, funcs, calls = get_np_arrays(search_df)
        results[f"{package_name}_sim_annealing"] = OptResult(
            "sim_annealing",
            search_df,
            epochs,
            best_score,
            replicates=replicates,
            permutates=permutates,
        )
        record["sim_annealing_search_dwm"] = best_score
        record["sim_annealing_search_m"] = get_modularity_score(
            modules, classes, funcs, calls, modularity
        )

    if use_genetic_search:
        search_df, epochs, best_score = genetic_search(
            raw_calls_df,
            module_names=module_names,
            class_names=class_names,
            func_names=func_names,
            population=pop,
            sims=sims,
        )

        modules, classes, funcs, calls = get_np_arrays(search_df)
        results[f"{package_name}_genetic_search"] = OptResult(
            "genetic_search",
            search_df,
            epochs,
            best_score,
            replicates=replicates,
            permutates=permutates,
        )
        record["genetic_search_dwm"] = best_score
        record["genetic_search_m"] = get_modularity_score(
            modules, classes, funcs, calls, modularity
        )

    print("*" * 79, end="\n\n")
    return record, results
