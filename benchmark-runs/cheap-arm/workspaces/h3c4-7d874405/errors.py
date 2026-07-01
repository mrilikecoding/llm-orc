class ParseError(ValueError):
    """Raised when text cannot be parsed as an integer."""
    pass


class RangeError(ValueError):
    """Raised when a value falls outside the allowed bounds."""
    pass
