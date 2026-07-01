from errors import RangeError


def in_range(value, low, high):
    if value < low or value > high:
        raise RangeError(
            f"Value {value} is not in range [{low}, {high}]"
        )
    return value
