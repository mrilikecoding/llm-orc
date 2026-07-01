from errors import ParseError

def parse_int(text):
    try:
        return int(text)
    except ValueError:
        raise ParseError("Non-numeric input") from None