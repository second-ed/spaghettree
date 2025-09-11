def func_d() -> int:
    return 3


class ClassA:
    def method_a(self) -> int:
        return func_d() + func_d()
