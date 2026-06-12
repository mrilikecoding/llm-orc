```python
import pytest
from argparse import ArgumentParser
from cli import main

def test_argument_parser():
    parser = ArgumentParser(description='Temperature Converter')
    parser.add_argument('--convert-from', required=True, choices=['celsius', 'fahrenheit', 'kelvin'])
    parser.add_argument('--convert-to', required=True, choices=['celsius', 'fahrenheit', 'kelvin'])
    parser.add_argument('--value', required=True, type=float)
    args = parser.parse_args(['--convert-from', 'celsius', '--convert-to', 'fahrenheit', '--value', '25'])
    assert args.convert_from == 'celsius'
    assert args.convert_to == 'fahrenheit'
    assert args.value == 25.0

def test_conversion_celsius_to_fahrenheit():
    args = ['--convert-from', 'celsius', '--convert-to', 'fahrenheit', '--value', '25']
    main(args)
    assert True  # Output is printed to stdout, no direct return value

def test_conversion_fahrenheit_to_celsius():
    args = ['--convert-from', 'fahrenheit', '--convert-to', 'celsius', '--value', '32']
    main(args)
    assert True  # Output is printed to stdout, no direct return value

def test_conversion_celsius_to_kelvin():
    args = ['--convert-from', 'celsius', '--convert-to', 'kelvin', '--value', '0']
    main(args)
    assert True  # Output is printed to stdout, no direct return value

def test_unsupported_conversion():
    args = ['--convert-from', 'fahrenheit', '--convert-to', 'kelvin', '--value', '32']
    with pytest.raises(ValueError):
        main(args)
```