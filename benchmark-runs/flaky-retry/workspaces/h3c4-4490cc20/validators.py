from errors import RangeError

def in_range(value, low, high):
    if low <= value <= high:
        return value
    else:
        raise RangeError(f"Value {value} is not in range [{low}, {high}].")