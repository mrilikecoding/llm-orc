```python
import argparse
import converters

def main():
    parser = argparse.ArgumentParser(description='Temperature conversion CLI.')
    parser.add_argument('value', type=float, help='Temperature value to convert.')
    parser.add_argument('conversion', choices=['celsius_to_fahrenheit', 'fahrenheit_to_celsius', 'celsius_to_kelvin'], help='Conversion type.')
    args = parser.parse_args()

    if args.conversion == 'celsius_to_fahrenheit':
        result = converters.celsius_to_fahrenheit(args.value)
    elif args.conversion == 'fahrenheit_to_celsius':
        result = converters.fahrenheit_to_celsius(args.value)
    elif args.conversion == 'celsius_to_kelvin':
        result = converters.celsius_to_kelvin(args.value)
    else:
        parser.error("Invalid conversion type")

    print(f"Result: {result}")

if __name__ == '__main__':
    main()
```

```python
import pytest
import converters

def test_celsius_to_fahrenheit(celsius, expected):
    assert round(converters.celsius_to_fahrenheit(celsius), 2) == round(expected, 2)

def test_fahrenheit_to_celsius(fahrenheit, expected):
    assert round(converters.fahrenheit_to_celsius(fahrenheit), 2) == round(expected, 2)

def test_celsius_to_kelvin(celsius, expected):
    assert round(converters.celsius_to_kelvin(celsius), 2) == round(expected, 2)

@pytest.mark.parametrize("celsius, expected", [
    (0, 32.0),
    (100, 212.0),
    (-40, -40.0),
    (25, 77.0),
])
def test_celsius_to_fahrenheit_valid(celsius, expected):
    test_celsius_to_fahrenheit(celsius, expected)

@pytest.mark.parametrize("fahrenheit, expected", [
    (32, 0.0),
    (212, 100.0),
    (-40, -40.0),
    (77, 25.0),
])
def test_fahrenheit_to_celsius_valid(fahrenheit, expected):
    test_fahrenheit_to_celsius(fahrenheit, expected)

@pytest.mark.parametrize("celsius, expected", [
    (0, 273.15),
    (100, 373.15),
    (-273.15, 0.0),
    (25, 298.15),
])
def test_celsius_to_kelvin_valid(celsius, expected):
    test_celsius_to_kelvin(celsius, expected)

def test_celsius_to_fahrenheit_error_handling():
    with pytest.raises(TypeError):
        converters.celsius_to_fahrenheit("invalid")

def test_fahrenheit_to_celsius_error_handling():
    with pytest.raises(TypeError):
        converters.fahrenheit_to_celsius("invalid")

def test_celsius_to_kelvin_error_handling():
    with pytest.raises(TypeError):
        converters.celsius_to_kelvin("invalid")
```

**Critical Issues Identified:**

1. **Circular Import in `converters.py`:**  
   The `converters.py` files in both `ws_2` and `ws_3` contain the line `import converters`, which creates a **circular import**. This causes a runtime error when the module is loaded because the module is importing itself.

2. **Redundant CLI Code in `converters.py`:**  
   The `converters.py` files incorrectly include the CLI logic (from `cli.py`) such as `argparse` setup and command-line argument parsing. This violates the separation of concerns—`converters.py` should **only** contain the temperature conversion functions, not CLI code.

3. **Incorrect File Structure:**  
   The `converters.py` files are mixing **function definitions** with **CLI logic**, which is not required. The CLI code should be in a separate `cli.py` file as originally provided.

**Recommendations:**  
- Remove the `import converters` line from `converters.py` to eliminate the circular import.  
- Move all CLI logic (e.g., `argparse` setup) to a dedicated `cli.py` file.  
- Ensure `converters.py` contains **only** the three temperature conversion functions.