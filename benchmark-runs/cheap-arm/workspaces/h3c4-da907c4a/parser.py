from errors import ParseError


def parse_int(text):
    try:
        return int(text)
    except (ValueError, TypeError) as exc:
        raise ParseError(f"cannot parse '{text}' as integer") from exc
