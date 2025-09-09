from functools import partial

from spaghettree import Result
from spaghettree.adapters.io_wrapper import IOProtocol, IOWrapper
from spaghettree.domain.adj_mat import AdjMat
from spaghettree.domain.optimisation import (
    merge_single_entity_communities_if_no_gain_penalty,
    optimise_communities,
)
from spaghettree.domain.parsing import (
    create_call_tree,
    extract_entities_and_locations,
    filter_non_native_calls,
    pair_exclusive_calls,
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
from spaghettree.logger import logger


def main(src_root: str, new_root: str) -> Result:
    io = IOWrapper()
    return run_process(io, src_root, new_root)


def run_process(io: IOProtocol, src_root: str, new_root: str) -> Result:
    logger.info(f"*** RUNNING `spaghettree` {src_root = } {new_root = } ***")
    ent_and_locs_res = io.read_files(src_root).and_then(
        partial(extract_entities_and_locations, root=src_root)
    )

    if not ent_and_locs_res.is_ok():
        raise ent_and_locs_res.error

    entities, location_map = ent_and_locs_res.inner

    entities_res = filter_non_native_calls(entities)

    if not entities_res.is_ok():
        raise entities_res.error

    entities = entities_res.inner

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
                order_map=location_map,
            ),
        )
        .and_then(partial(create_new_filepaths, new_root=new_root or src_root))
        .and_then(add_empty_inits_if_needed)
        .and_then(partial(io.write_files, ruff_root=new_root or src_root))
    )
