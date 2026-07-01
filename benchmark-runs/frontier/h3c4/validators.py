from errors import RangeError


def in_range(value: int, low: int, high: int) -> int:
    if value < low or value > high:
        raise RangeError(f"{value} is not in range [{low}, {high}]")
    return value
