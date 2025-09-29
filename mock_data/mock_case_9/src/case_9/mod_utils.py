from typing import TypeVar

T = TypeVar('T')


def process_data[T](data: list[T]) -> list[T]:
    return [item for item in data if item is not None]


def get_first_item[T](items: list[T]) -> T:
    return items[0]
