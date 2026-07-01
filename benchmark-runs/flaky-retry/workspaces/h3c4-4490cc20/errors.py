class ParseError(Exception):
    def __init__(self, message):
        super().__init__(message)

class RangeError(Exception):
    def __init__(self, message):
        super().__init__(message)