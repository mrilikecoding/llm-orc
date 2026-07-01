from errors import RangeError

def in_range(value, low, high):
    if low > high:
        raise RangeError("Low bound cannot exceed high bound")
    if value < low or value > high:
        raise RangeError
    return value