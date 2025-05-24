from __future__ import annotations

import glob
from pathlib import Path

import black
import isort
import yaml
from returns.result import Failure, Result, Success, safe


@safe
def read_file(path: str) -> str:
    with open(path, "r") as f:
        data = f.read()
    return data


def read_files(root: str | Path) -> Result[dict[str, str], Exception]:
    paths = glob.glob(f"{root}/**/**.py", recursive=True)

    results = {}
    for path in paths:
        res = read_file(path)
        match res:
            case Success(data):
                results[path] = data
            case Failure(_):
                print(res)
                return res
    return Success(results)


def write_file(modified_code: str, filepath: str, format_code: bool = True) -> Result[bool, Exception]:
    def format_code_str(code_snippet: str) -> str:
        return black.format_str(isort.code(code_snippet), mode=black.FileMode())

    try:
        if format_code:
            modified_code = format_code_str(modified_code)
        with open(filepath, "w") as f:
            f.write(modified_code)
        return Success(True)
    except Exception as e:
        print(f"{e} for {filepath}")
        return Failure(e)


def read_yaml(path: str) -> dict:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data
