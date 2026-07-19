def divide(a, b):
    if b == 0:
        raise ValueError("cannot divide by zero")
    return a / b


def percent(part, whole):
    return divide(part, whole) * 100
