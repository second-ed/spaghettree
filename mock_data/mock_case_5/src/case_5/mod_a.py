import attrs


@attrs.define
class SomeClass:
    name: str = attrs.field()

    def method_a(self) -> str:
        return self.name.upper()
