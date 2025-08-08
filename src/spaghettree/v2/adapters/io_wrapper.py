import glob
import subprocess
from pathlib import Path

import attrs

from spaghettree.v2 import Ok, Result, safe


@attrs.define
class IOWrapper:
    @safe
    def list_files(self, root: str | Path, recursive: bool = True) -> list[str]:
        return glob.glob(f"{root}/**/**.py", recursive=recursive)

    @safe
    def read(self, path: str) -> str:
        with open(path, "r") as f:
            data = f.read()
        return data

    def read_files(self, root: str | Path) -> Result:
        paths_res = self.list_files(root)
        if not paths_res.is_ok():
            return paths_res
        paths = paths_res.inner

        results, fails = {}, {}
        for path in paths:
            res = self.read(path)
            if res.is_ok():
                results[path] = res.inner
            else:
                fails[path] = res

        return Ok(results)

    @safe
    def write(self, modified_code: str, filepath: str, format_code: bool = True) -> None:
        with open(filepath, "w") as f:
            f.write(modified_code)
        if format_code:
            subprocess.run(["ruff", "check", "--fix", str(filepath)], check=True)
            subprocess.run(["ruff", "format", str(filepath)], check=True)
