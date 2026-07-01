class ParseError(ValueError):
    """Raised when input cannot be parsed as expected."""
    pass


class RangeError(ValueError):
    """Raised when a value falls outside acceptable bounds."""
    pass
