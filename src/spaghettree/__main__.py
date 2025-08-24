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
    add_empty_inits_if_needed,
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
        .and_then(create_module_cst_objs)
        .and_then(resolve_module_calls)
        .and_then(extract_entities)
        .and_then(filter_non_native_calls)
    )

    if entities_res.is_ok():
        entities = entities_res.inner
    else:
        raise entities_res.error

    return (
        entities_res.and_then(create_call_tree)
        .and_then(AdjMat.from_call_tree)
        .and_then(pair_exclusive_calls)
        .and_then(optimise_communities)
        .and_then(merge_single_entity_communities_if_no_gain_penalty)
        .and_then(partial(create_new_module_map, entities=entities))
        .and_then(infer_module_names)
        .and_then(rename_overlapping_mod_names)
        .and_then(remap_imports)
        .and_then(
            partial(
                convert_to_code_str,
                type_priority={
                    "GlobalCST": 0,
                    "ClassCST": 1,
                    "FuncCST": 2,
                },
            ),
        )
        .and_then(partial(create_new_filepaths, src_root=new_root))
        .and_then(add_empty_inits_if_needed)
        .and_then(partial(io.write_files, ruff_root=new_root))
    )
