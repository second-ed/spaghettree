from functools import partial

from spaghettree import Ok, Result
from spaghettree.adapters.io_wrapper import IOProtocol, IOWrapper
from spaghettree.domain.adj_mat import AdjMat
from spaghettree.domain.entities import EntityCST
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
from spaghettree.domain.visitors import EntityLocation
from spaghettree.logger import logger


def main(src_root: str, *, new_root: str = "", optimise_src_code: bool = True) -> Result:
    io = IOWrapper()
    return run_process(io, src_root, new_root=new_root, optimise_src_code=optimise_src_code)


def run_process(
    io: IOProtocol, src_root: str, *, new_root: str = "", optimise_src_code: bool = True
) -> Result:
    logger.info(f"*** RUNNING `spaghettree` {src_root = } {new_root = } ***")
    src_code = io.read_files(src_root).unwrap()

    ent_and_locs_res = extract_entities_and_locations(src_code)
    entities, location_map = ent_and_locs_res.unwrap()
    entities_res = filter_non_native_calls(entities)
    entities = entities_res.unwrap()
    call_tree = entities_res.and_then(create_call_tree).unwrap()

    if optimise_src_code:
        return optimise_entity_positions(
            io=io,
            entities=entities,
            location_map=location_map,
            call_tree=call_tree,
            src_root=src_root,
            new_root=new_root,
        )

    return Ok(call_tree)


def optimise_entity_positions(  # noqa: PLR0913
    io: IOProtocol,
    entities: dict[str, EntityCST],
    location_map: dict[str, EntityLocation],
    call_tree: dict[str, list[str]],
    src_root: str,
    new_root: str,
) -> Result:
    return (
        AdjMat.from_call_tree(call_tree)
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
