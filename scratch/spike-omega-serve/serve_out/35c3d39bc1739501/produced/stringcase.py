def snake_to_camel(s: str) -> str:
    parts = s.split('_')
    return ''.join(part.capitalize() for part in parts)

def camel_to_snake(s: str) -> str:
    result = []
    for i, c in enumerate(s):
        if c.isupper():
            if i == 0:
                result.append(c.lower())
            else:
                result.append('_')
                result.append(c.lower())
        else:
            result.append(c)
    return ''.join(result)