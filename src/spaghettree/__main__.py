from functools import partial

from spaghettree import Result
from spaghettree.adapters.io_wrapper import IOWrapper
from spaghettree.domain.adj_mat import AdjMat
from spaghettree.domain.optimisation import (
    merge_single_entity_communities_if_no_gain_penalty,
    optimise_communities,
)
from spaghettree.domain.parsing import (
    create_call_tree,
    create_module_cst_objs,
    extract_entities,
    filter_non_native_calls,
    pair_exclusive_calls,
    resolve_module_calls,
)
from spaghettree.domain.processing import (
    convert_to_code_str,
    create_new_filepaths,
    create_new_module_map,
    infer_module_names,
    remap_imports,
    rename_overlapping_mod_names,
)


def main(src_root: str, new_root: str) -> Result:
    io = IOWrapper()
    entities_res = (
        io.read_files(src_root)
        .bind(create_module_cst_objs)
        .bind(resolve_module_calls)
        .bind(extract_entities)
        .bind(filter_non_native_calls)
    )

    entities = entities_res.inner

    return (
        entities_res.bind(create_call_tree)
        .bind(AdjMat.from_call_tree)
        .bind(pair_exclusive_calls)
        .bind(optimise_communities)
        .bind(merge_single_entity_communities_if_no_gain_penalty)
        .bind(partial(create_new_module_map, entities=entities))
        .bind(infer_module_names)
        .bind(rename_overlapping_mod_names)
        .bind(remap_imports)
        .bind(
            partial(
                convert_to_code_str,
                type_priority={
                    "GlobalCST": 0,
                    "ClassCST": 1,
                    "FuncCST": 2,
                },
            )
        )
        .bind(partial(create_new_filepaths, src_root=new_root))
        .bind(partial(io.write_files, ruff_root=new_root))
    )
