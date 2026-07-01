from errors import ParseError


def parse_int(text: str) -> int:
    try:
        return int(text)
    except (ValueError, TypeError) as e:
        raise ParseError(f"Cannot parse {text!r} as int") from e
