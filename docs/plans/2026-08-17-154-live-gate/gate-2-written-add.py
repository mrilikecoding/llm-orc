def add(a, b):
    if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        raise TypeError("Both inputs must be numbers")
    return a + b
def test_add_string_raises_error():
       try:
           add("a", 2)
       except TypeError:
           assert True, "Expected TypeError"
       else:
           assert False, "Did not raise TypeError"
def add(a, b):
    if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        raise TypeError("Both inputs must be numbers")
    return a + b