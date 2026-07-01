from parser import parse_int
from validators import in_range


def run(text: str, low: int, high: int) -> int:
    value = parse_int(text)
    return in_range(value, low, high)
