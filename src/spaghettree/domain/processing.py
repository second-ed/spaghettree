import os
from collections import Counter, defaultdict
from copy import deepcopy
from functools import partial

from spaghettree import safe
from spaghettree.domain.adj_mat import AdjMat
from spaghettree.domain.imports import ImportCST
from spaghettree.domain.parsing import EntityCST, cst_to_str


@safe
def create_new_module_map(
    adj_mat: AdjMat,
    entities: dict[str, EntityCST],
) -> dict[int, list[EntityCST]]:
    new_modules: defaultdict[int, list[EntityCST]] = defaultdict(list)

    for i, mod_name in enumerate(adj_mat.communities):
        ent_name = adj_mat.node_map[i]
        new_modules[mod_name].append(entities[ent_name])

    return dict(new_modules)


@safe
def infer_module_names(
    new_modules: dict[int, list[EntityCST]],
) -> dict[str, list[EntityCST]]:
    renamed_modules: dict[str, list[EntityCST]] = {}

    for contents in new_modules.values():
        if len(contents) > 1:
            names = [".".join(ent.name.split(".")[:-1]) for ent in contents]
            possible_module_names = sorted(
                {(name, names.count(name)) for name in names},
                key=lambda x: x[1],
                reverse=True,
            )
            for name, _ in possible_module_names:
                if name not in renamed_modules:
                    mod_name = name
                    break
            else:
                mod_name = f"{possible_module_names[0][0]}.mod_overflow"
        else:
            mod_name = contents[0].name
        renamed_modules[mod_name] = contents
    return renamed_modules


@safe
def rename_overlapping_mod_names(
    renamed_modules: dict[str, list[EntityCST]],
) -> dict[str, list[EntityCST]]:
    def rename_mod_name(name: str, renamed_modules: list[str]) -> str:
        name_parts = name.split(".")
        dirname = ".".join(name_parts[:-1])

        dirnames = [".".join(m.split(".")[:-1]) for m in renamed_modules]
        dirname_counts = Counter(dirnames)

        if dirname not in renamed_modules and dirname_counts.get(dirname, 0) <= 1:
            return dirname

        if dirname in renamed_modules:
            return ".".join([*name_parts[:-2], "_".join(name_parts[-2:])])

        return name

    mod_names = list(renamed_modules)
    return {
        rename_mod_name(name, mod_names): contents for name, contents in renamed_modules.items()
    }


@safe
def create_new_filepaths(
    fixed_name_modules: dict[str, list[EntityCST]],
    src_root: str,
) -> dict[str, list[EntityCST]]:
    def to_filepath(src_root: str, name: str) -> str:
        return os.path.join(os.path.dirname(src_root), name.replace(".", "/") + ".py")

    return {to_filepath(src_root, name): contents for name, contents in fixed_name_modules.items()}


@safe
def convert_to_code_str(
    new_modules: dict[str, list[EntityCST]],
    type_priority: dict[str, int],
) -> dict[str, str]:
    def get_module_str(mod_contents: list[EntityCST]) -> str:
        imports, code = [], []

        for ent in mod_contents:
            imports.extend([imp.to_str() for imp in ent.imports])
            code.append(cst_to_str(ent.tree))

        return "".join(sorted(set(imports))) + "".join(code)

    def sort_by_priority(
        contents: list[EntityCST],
        type_priority: dict[str, int],
    ) -> list[EntityCST]:
        def sort_key(obj: EntityCST, type_priority: dict[str, int]) -> tuple[int, str]:
            return (type_priority[obj.__class__.__name__], getattr(obj, "name", ""))

        return sorted(contents, key=partial(sort_key, type_priority=type_priority))

    return {
        mod_name: get_module_str(sort_by_priority(contents, type_priority))
        for mod_name, contents in new_modules.items()
    }


@safe
def remap_imports(
    modules: dict[str, list[EntityCST]],
) -> dict[str, list[EntityCST]]:
    modules = deepcopy(modules)
    entity_mod_map: dict[str, str] = {
        ent.name: mod_name for mod_name, ents in modules.items() for ent in ents
    }

    for mod_name, ents in modules.items():
        for ent in ents:
            updated_imports: list[ImportCST] = []

            for imp in ent.imports:
                new_mod = entity_mod_map.get(f"{imp.module}.{imp.name}")
                if new_mod is None:
                    updated_imports.append(imp)
                elif new_mod != mod_name:
                    updated_imports.append(
                        ImportCST(
                            module=new_mod,
                            import_type=imp.import_type,
                            name=imp.name,
                            as_name=imp.as_name,
                        ),
                    )

            ent.imports = updated_imports
    return modules


@safe
def add_empty_inits_if_needed(modules: dict[str, str]) -> dict[str, str]:
    modules_with_inits = {}

    for path, contents in modules.items():
        init_path = f"{os.path.dirname(path)}/__init__.py"

        if init_path not in modules:
            modules_with_inits[init_path] = ""
        modules_with_inits[path] = contents

    return modules_with_inits
