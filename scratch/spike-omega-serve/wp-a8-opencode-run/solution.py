def pad_string(s, width, pad_char=' '):
    """
    Pad a string to a fixed width with a specified character.
    
    Args:
        s: Input string
        width: Target width (minimum length)
        pad_char: Character to use for padding (default: space)
        
    Returns:
        Padded string (unchanged if original length >= width)
    """
    if len(s) >= width:
        return s
    return s + pad_char * (width - len(s))