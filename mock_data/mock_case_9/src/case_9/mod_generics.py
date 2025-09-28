from typing import Optional

from case_9.mod_utils import process_data


def process_list(items: list[str]) -> list[str]:
    return process_data(items)


def create_mapping(keys: list[str], values: list[int]) -> dict[str, int]:
    result: dict[str, int] = {}
    for key, value in zip(keys, values):
        result[key] = value
    return result


def find_item(items: list[str], target: str) -> str | None:
    for item in items:
        if item == target:
            return item
    return None
