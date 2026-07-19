def mean(values):
    if not values:
        raise ValueError("no values")
    return sum(values) / len(values)
