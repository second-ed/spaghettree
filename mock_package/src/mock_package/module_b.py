def func_e(a: int, b: int) -> int:
    return a + b + func_d()


def func_d():
    return -2


class ClassA:
    def method_a(self):
        return func_d()
