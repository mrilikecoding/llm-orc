class ParseError(Exception):
    """Raised when text cannot be parsed as an integer."""
    pass


class RangeError(Exception):
    """Raised when a value falls outside the allowed range."""
    pass
