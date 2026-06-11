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

```python
import subprocess
import pytest

def test_cli_valid_conversion():
    result = subprocess.run(
        ["python", "cli.py", "25", "celsius_to_fahrenheit"],
        capture_output=True,
        text=True,
        check=True
    )
    assert "Result: 77.0" in result.stdout

def test_cli_fahrenheit_to_celsius_valid():
    result = subprocess.run(
        ["python", "cli.py", "32", "fahrenheit_to_celsius"],
        capture_output=True,
        text=True,
        check=True
    )
    assert "Result: 0.0" in result.stdout

def test_cli_celsius_to_kelvin_valid():
    result = subprocess.run(
        ["python", "cli.py", "25", "celsius_to_kelvin"],
        capture_output=True,
        text=True,
        check=True
    )
    assert "Result: 298.15" in result.stdout

def test_cli_invalid_conversion():
    result = subprocess.run(
        ["python", "cli.py", "25", "invalid_conversion"],
        capture_output=True,
        text=True,
        check=False
    )
    assert "error: argument conversion: invalid choice: 'invalid_conversion' (choose from 'celsius_to_fahrenheit', 'fahrenheit_to_celsius', 'celsius_to_kelvin')" in result.stderr

def test_cli_missing_arguments():
    result = subprocess.run(
        ["python", "cli.py"],
        capture_output=True,
        text=True,
        check=False
    )
    assert "error: the following arguments are required: value, conversion" in result.stderr

def test_cli_help():
    result = subprocess.run(
        ["python", "cli.py", "--help"],
        capture_output=True,
        text=True,
        check=True
    )
    assert "usage: cli.py [-h] value conversion" in result.stdout
    assert "Temperature conversion CLI." in result.stdout
    assert "conversion            Conversion type." in result.stdout
    assert "choices: celsius_to_fahrenheit, fahrenheit_to_celsius, celsius_to_kelvin" in result.stdout

def test_cli_non_numeric_value():
    result = subprocess.run(
        ["python", "cli.py", "abc", "celsius_to_fahrenheit"],
        capture_output=True,
        text=True,
        check=False
    )
    assert "invalid literal for float() with base 10: 'abc'" in result.stderr
```