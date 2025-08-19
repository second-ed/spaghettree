def func_e(a: int, b: int) -> int:  # noqa: INP001
    return a + b + func_d()


def func_d():  # noqa: ANN201
    return -2


class ClassA:
    def method_a(self):  # noqa: ANN201
        return func_d() + func_d()
