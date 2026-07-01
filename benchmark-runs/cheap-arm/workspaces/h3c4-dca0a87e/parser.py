from errors import ParseError


def parse_int(text):
    try:
        return int(text)
    except (ValueError, TypeError) as e:
        raise ParseError(f"could not parse integer from {text!r}") from e
