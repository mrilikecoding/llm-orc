import pytest
import sys

def test_celsius_to_fahrenheit_single():
    sys.argv = ['converters', 'celsius_to_fahrenheit', '25']
    main()
    assert "25 Celsius to Fahrenheit: 77.0" in sys.stdout.getvalue()

def test_celsius_to_fahrenheit_multiple():
    sys.argv = ['converters', 'celsius_to_fahrenheit', '25', '30', '35']
    main()
    output = sys.stdout.getvalue()
    assert "25 Celsius to Fahrenheit: 77.0" in output
    assert "30 Celsius to Fahrenheit: 86.0" in output
    assert "35 Celsius to Fahrenheit: 95.0" in output

def test_fahrenheit_to_celsius_single():
    sys.argv = ['converters', 'fahrenheit_to_celsius', '77']
    main()
    assert "77 Fahrenheit to Celsius: 25.0" in sys.stdout.getvalue()

def test_celsius_to_kelvin_single():
    sys.argv = ['converters', 'celsius_to_kelvin', '25']
    main()
    assert "25 Celsius to Kelvin: 298.15" in sys.stdout.getvalue()

def test_invalid_function():
    sys.argv = ['converters', 'invalid_function', '25']
    main()
    assert "Unknown function: invalid_function" in sys.stderr.getvalue()

def test_missing_arguments():
    sys.argv = ['converters', 'celsius_to_fahrenheit']
    main()
    assert "Usage: python -m converters <function> <temperature> ..." in sys.stderr.getvalue()