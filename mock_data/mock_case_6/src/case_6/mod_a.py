import math

PI: float = math.pi


class Circle:
    def __init__(self, radius: float) -> None:
        self.radius = radius

    def calc_area(self) -> float:
        return PI * self.radius * self.radius
