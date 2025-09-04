CONSTANT = 0


def func_e(a: int, b: int) -> int:
    return a + b + func_d() + CONSTANT


def func_d():  # noqa: ANN201
    return -2


class ClassA:
    def method_a(self):  # noqa: ANN201
        return func_d() + func_d()
