from __future__ import annotations

import datetime as dt
import os

import attrs
import pandas as pd
import yaml
from returns.result import Failure, Result, Success

from spaghettree.utils import format_code_str


def get_src_code(path: str) -> Result[str, Exception]:
    try:
        with open(path, "r") as f:
            src_code = f.read()
        return Success(src_code)
    except Exception as e:
        print(f"{e} for {path}")
        return Failure(e)


def save_modified_code(
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
