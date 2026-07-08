def pad_string(s, width):
    if len(s) > width:
        return s
    return s.ljust(width)