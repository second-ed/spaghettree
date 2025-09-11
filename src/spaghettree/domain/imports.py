from __future__ import annotations

from enum import Enum, auto

import attrs
from attrs.validators import instance_of


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
