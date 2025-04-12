from __future__ import annotations

import glob
import os
import pprint

from spaghettree.data_structures import OptResult
from spaghettree.metrics import modularity_df
from spaghettree.processing import (
    clean_calls_df,
    get_call_table,
    get_entity_names,
    get_modules,
)
from spaghettree.search import genetic_search, get_modularity_score, hill_climber_search


def process_package(p: str, total_sims: int = 8000, pop: int = 8):
    sims = total_sims // pop
    package_name = os.path.basename(p)
    # print("*" * 79)
    # print(f"{package_name = }")
    paths = glob.glob(f"{p}/**/**.py", recursive=True)
    modules = get_modules(paths)
    module_names, func_names, class_names = modules.bind(get_entity_names)
    raw_calls = modules.bind(get_call_table)
    raw_calls_df = raw_calls.unwrap()
    calls = raw_calls.bind(clean_calls_df)

    record = {
        "package_name": package_name,
        "n_modules": len(module_names),
        "n_classes": len(class_names),
        "n_funcs": len(func_names),
        "base_dwm": get_modularity_score(calls.unwrap()),
        "base_modularity": get_modularity_score(calls.unwrap(), modularity_df),
        "total_sims": total_sims,
        "initial_population_size": pop,
        "generations": sims,
    }
    pprint.pprint(record, sort_dicts=False)

    search_df, epochs, best_score = hill_climber_search(
        raw_calls_df,
        module_names=module_names,
        class_names=class_names,
        func_names=func_names,
        sims=total_sims,
    )
    record["hill_climber_search_dwm"] = best_score
    record["hill_climber_search_m"] = get_modularity_score(search_df, modularity_df)

    search_df, epochs, best_score = genetic_search(
        raw_calls_df, module_names, class_names, func_names, population=pop, sims=sims
    )
    record["genetic_search_dwm"] = best_score
    record["genetic_search_m"] = get_modularity_score(search_df, modularity_df)

    # print("*" * 79, end="\n\n")
    return record, {
        f"{package_name}_hill_climber": OptResult(
            "hill_climber", search_df, epochs, best_score
        ),
        f"{package_name}_genetic_search": OptResult(
            "genetic_search", search_df, epochs, best_score
        ),
    }
