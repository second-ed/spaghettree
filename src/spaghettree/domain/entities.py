from __future__ import annotations

from collections.abc import Collection
from enum import Enum, auto
from typing import Protocol, Self, runtime_checkable

import attrs
import libcst as cst
from attrs.validators import instance_of


@runtime_checkable
class EntityCST(Protocol):
    def get_call_tree_entries(self) -> list[str]: ...

    def resolve_calls(self, import_map: dict[str, str], ent_map: dict[str, str]) -> Self: ...

    def filter_native_calls(self, entities: Collection[str]) -> Self: ...

    def resolve_native_imports(self) -> Self: ...

    def add_referenced_imports(self, imports: set[ImportCST]) -> Self: ...


@attrs.define
class ClassCST:
    name: str = attrs.field(validator=[instance_of(str)])
    tree: cst.ClassDef = attrs.field(validator=[instance_of(cst.ClassDef)], repr=False)
    methods: list[FuncCST] = attrs.field(factory=list, validator=[instance_of(list)])
    imports: set[ImportCST] = attrs.field(factory=set)
    bases: list[str] = attrs.field(factory=list)

    def get_call_tree_entries(self) -> list[str]:
        return [call for meth in self.methods for call in meth.calls]

    def resolve_calls(self, import_map: dict[str, str], ent_map: dict[str, str]) -> Self:
        # piggyback here to get the subclass relationships
        self.bases = resolve_calls(self.bases, import_map, ent_map)

        for meth in self.methods:
            meth.resolve_calls(import_map, ent_map)
        return self

    def filter_native_calls(self, entities: Collection[str]) -> Self:
        for meth in self.methods:
            bases_and_calls = self.bases + meth.calls
            meth.calls = [
                call for call in bases_and_calls if call in entities and meth != self.name
            ]
        return self

    def resolve_native_imports(self) -> Self:
        for method in self.methods:
            for call in method.calls:
                call_parts = call.split(".")
                mod_name = ".".join(call_parts[:-1])
                call_name = call_parts[-1]
                self.imports.add(ImportCST(mod_name, ImportType.FROM, call_name, call_name))
        return self

    def add_referenced_imports(self, imports: set[ImportCST]) -> Self:
        for imp in imports:
            for meth in self.methods:
                if imp.as_name in meth.calls or f"{imp.module}.{imp.as_name}" in meth.calls:
                    self.imports.add(imp)


@attrs.define
class FuncCST:
    name: str = attrs.field(validator=[instance_of(str)])
    tree: cst.FunctionDef = attrs.field(validator=[instance_of(cst.FunctionDef)], repr=False)
    calls: list[str] = attrs.field(factory=list, validator=[instance_of(list)])
    imports: set[ImportCST] = attrs.field(factory=set)

    def get_call_tree_entries(self) -> list[str]:
        return self.calls

    def resolve_calls(self, import_map: dict[str, str], ent_map: dict[str, str]) -> Self:
        self.calls = resolve_calls(self.calls, import_map, ent_map)
        return self

    def filter_native_calls(self, entities: Collection[str]) -> Self:
        self.calls = [call for call in self.calls if call in entities and call != self.name]
        return self

    def resolve_native_imports(self) -> Self:
        for call in self.calls:
            call_parts = call.split(".")
            mod_name = ".".join(call_parts[:-1])
            call_name = call_parts[-1]
            self.imports.add(ImportCST(mod_name, ImportType.FROM, call_name, call_name))
        return self

    def add_referenced_imports(self, imports: set[ImportCST]) -> Self:
        for imp in imports:
            if imp.as_name in self.calls or f"{imp.module}.{imp.as_name}" in self.calls:
                self.imports.add(imp)
        return self


@attrs.define(eq=True)
class GlobalCST:
    name: str = attrs.field()
    tree: cst.SimpleStatementLine = attrs.field(repr=False)
    referenced: list[str] = attrs.field(factory=list)
    imports: set[ImportCST] = attrs.field(factory=set)

    def get_call_tree_entries(self) -> list[str]:
        return self.referenced

    def resolve_calls(self, import_map: dict[str, str], ent_map: dict[str, str]) -> Self:
        self.referenced = resolve_calls(self.referenced, import_map, ent_map)
        return self

    def filter_native_calls(self, entities: Collection[str]) -> Self:
        self.referenced = [ref for ref in self.referenced if ref in entities and ref != self.name]
        return self

    def resolve_native_imports(self) -> Self:
        for ref in self.referenced:
            ref_parts = ref.split(".")
            mod_name = ".".join(ref_parts[:-1])
            ref_name = ref_parts[-1]
            self.imports.add(ImportCST(mod_name, ImportType.FROM, ref_name, ref_name))
        return self

    def add_referenced_imports(self, imports: set[ImportCST]) -> Self:
        if self.name.endswith(".__all__"):
            self.imports.update(imports)
            return self

        for imp in imports:
            if imp.as_name in self.referenced:
                self.imports.add(imp)
        return self


class ImportType(Enum):
    FROM = auto()
    IMPORT = auto()


@attrs.define(frozen=True)
class ImportCST:
    module: str = attrs.field(validator=[instance_of(str)], converter=str.lower)
    import_type: ImportType = attrs.field(validator=[instance_of(ImportType)])
    name: str = attrs.field(validator=[instance_of(str)])
    as_name: str = attrs.field(validator=[instance_of(str)])

    def to_str(self) -> str:
        output: list[str] = []
        if self.import_type is ImportType.FROM:
            output.append(f"from {self.module} import {self.name}")
        elif self.import_type is ImportType.IMPORT:
            output.append(f"import {self.module}")

        if self.name != self.as_name:
            output.append(f"as {self.as_name}")
        return " ".join(output) + "\n"


def resolve_calls(
    calls: list[str],
    import_map: dict[str, str],
    ent_map: dict[str, str],
) -> list[str]:
    resolved_calls: list[str] = []
    for call in calls:
        call_parts = call.split(".")

        if resolved_call := import_map.get(call_parts[0]):
            resolved_call_parts = resolved_call.split(".")
            if resolved_call_parts[-1] != call:
                common_removed = ".".join(resolved_call_parts[:-1])
                resolved_calls.append(f"{common_removed}.{call}".strip("."))
            else:
                resolved_calls.append(resolved_call)
        elif resolved_call := ent_map.get(call_parts[0]):
            resolved_calls.append(resolved_call)
        else:
            resolved_calls.append(call)
    return resolved_calls
