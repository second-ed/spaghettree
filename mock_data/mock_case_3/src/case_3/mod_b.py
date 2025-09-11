from case_3.mod_a import A  # noqa: INP001

CONSTANT = 3_000


class B:
    def method_a(self) -> int:
        return CONSTANT


C = A | B
