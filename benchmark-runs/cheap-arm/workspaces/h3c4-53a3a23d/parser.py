from errors import ParseError


def parse_int(text):
    try:
        return int(text)
    except (ValueError, TypeError):
        raise ParseError(f"could not parse {text!r} as integer")
