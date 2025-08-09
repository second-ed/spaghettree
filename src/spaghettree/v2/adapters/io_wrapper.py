import glob
import os
import subprocess
from pathlib import Path

import attrs

from spaghettree.v2 import Err, Ok, Result, safe


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

        if fails:
            return Err(fails)
        return Ok(results)

    @safe
    def write(self, modified_code: str, filepath: str, format_code: bool = True) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            f.write(modified_code)
        if format_code:
            self._run_ruff(filepath)

    @safe
    def write_files(self, src_code: dict[str, str], ruff_root: str | None = None):
        results, fails = {}, {}

        for filepath, modified_code in src_code.items():
            if ruff_root is not None:
                # format all at the end instead
                res = self.write(modified_code, filepath, format_code=False)
            else:
                res = self.write(modified_code, filepath, format_code=True)

            if res.is_ok():
                results[filepath] = res.inner
            else:
                fails[filepath] = res

        if ruff_root:
            self._run_ruff(ruff_root)
        if fails:
            return Err(fails)
        return Ok(results)

    def _run_ruff(self, path: str):
        subprocess.run(["ruff", "check", "--fix", str(path)], check=True)
        subprocess.run(["ruff", "format", str(path)], check=True)
