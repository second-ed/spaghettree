from __future__ import annotations

from enum import Enum

import attrs
from attrs.validators import instance_of


class ImportType(Enum):
    FROM = "from"
    IMPORT = "import"


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

    def to_dict(self) -> dict:
        res = attrs.asdict(self)
        res["import_type"] = res["import_type"].value
        return res
