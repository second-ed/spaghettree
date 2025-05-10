from __future__ import annotations

import datetime as dt
import glob
import os

import attrs
import pandas as pd
import yaml
from returns.result import Failure, Result, Success, safe

from spaghettree.utils import format_code_str


@safe
def read_file(path: str) -> str:
    with open(path, "r") as f:
        data = f.read()
    return data


def read_files(root: str) -> dict[str, str]:
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


def write_file(
    modified_code: str, filepath: str, format_code: bool = True
) -> Result[bool, Exception]:
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


def write_yaml(data: dict, filepath: str) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def save_results(results: dict):
    now = dt.datetime.now().strftime(format="%y%m%d_%H%M")
    os.makedirs(f"./results/{now}", exist_ok=True)

    for name, result in results.items():
        package = result.package
        res_obj = attrs.asdict(result)

        for key, attrib in res_obj.items():
            if isinstance(attrib, pd.DataFrame):
                res_obj[key] = attrib.to_dict("records")

        os.makedirs(f"./results/{now}/{package}", exist_ok=True)
        write_yaml(res_obj, f"./results/{now}/{package}/{name}.yaml")
