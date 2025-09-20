from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable


@runtime_checkable
class MockProtocol(Protocol):
    def some_method(self, a: str) -> str: ...


class Base(ABC):
    @abstractmethod
    def some_method(self, a: str) -> str:
        pass


class Child(Base):
    def some_method(self, a: str) -> str:
        return a.upper()


class FreeClass:
    def some_other_method(self, a: int, b: int) -> int:
        return a * b
