CONSTANT = 0


def func_e(a: int, b: int) -> int:
    return a + b + func_d() + CONSTANT


def func_d() -> int:
    return 3


class ClassA:
    def method_a(self) -> int:
        return func_d() + func_d()
