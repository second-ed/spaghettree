from __future__ import annotations

from enum import Enum, auto

import attrs
import libcst as cst
from attrs.validators import instance_of


class ImportType(Enum):
    FROM = auto()
    IMPORT = auto()


@attrs.define(frozen=True)
class ImportCST:
    module: str = attrs.field(validator=[instance_of(str)])
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
        return " ".join(output)


@attrs.define
class ImportVisitor(cst.CSTVisitor):
    imports: list[ImportCST] = attrs.field(factory=list)

    def visit_Import(self, node: cst.Import) -> None:
        for alias in node.names:
            name = self._resolve_attr(alias.name)
            asname = alias.asname.name.value if alias.asname else name
            self._add_import(name, ImportType.IMPORT, name, asname)

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        module = self._resolve_attr(node.module)
        if module is None:
            return  # skip relative imports

        if isinstance(node.names, cst.ImportStar):
            self._add_import(module, ImportType.FROM, "*", "*")
            return

        aliases = node.names
        if isinstance(aliases, cst.ImportAlias):
            aliases = [aliases]

        for alias in aliases:
            name = self._resolve_attr(alias.name)
            asname = alias.asname.name.value if alias.asname else name
            self._add_import(module, ImportType.FROM, name, asname)

    def _add_import(self, key: str, import_type: ImportType, name: str, as_name: str) -> None:
        self.imports.append(ImportCST(key, import_type, name, as_name))

    def _resolve_attr(self, node: cst.BaseExpression | None) -> str | None:
        if node is None:
            return None
        if isinstance(node, cst.Name):
            return node.value
        if isinstance(node, cst.Attribute):
            parent = self._resolve_attr(node.value)
            return f"{parent}.{node.attr.value}" if parent else node.attr.value
        return None
