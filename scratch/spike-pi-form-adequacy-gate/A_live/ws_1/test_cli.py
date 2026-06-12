```python
import subprocess
import sys
import pytest

def test_convert_100_c_to_f():
    result = subprocess.run(
        [sys.executable, 'cli.py', '--value', '100', '--from', 'celsius', '--to', 'fahrenheit'],
        capture_output=True,
        text=True
    )
    assert result.stdout == "100 celsius is equal to 212.0 fahrenheit\n"

def test_convert_212_f_to_c():
    result = subprocess.run(
        [sys.executable, 'cli.py', '--value', '212', '--from', 'fahrenheit', '--to', 'celsius'],
        capture_output=True,
        text=True
    )
    assert result.stdout == "212 fahrenheit is equal to 100.0 celsius\n"

def test_convert_0_c_to_k():
    result = subprocess.run(
        [sys.executable, 'cli.py', '--value', '0', '--from', 'celsius', '--to', 'kelvin'],
        capture_output=True,
        text=True
    )
    assert result.stdout == "0 celsius is equal to 273.15 kelvin\n"

def test_invalid_conversion():
    result = subprocess.run(
        [sys.executable, 'cli.py', '--value', '32', '--from', 'fahrenheit', '--to', 'kelvin'],
        capture_output=True,
        text=True
    )
    assert result.stdout == "Unsupported conversion\n"

def test_missing_value():
    result = subprocess.run(
        [sys.executable, 'cli.py', '--from', 'celsius', '--to', 'fahrenheit'],
        capture_output=True,
        text=True
    )
    assert "error: the following arguments are required: --value" in result.stderr

def test_missing_from():
    result = subprocess.run(
        [sys.executable, 'cli.py', '--value', '100', '--to', 'fahrenheit'],
        capture_output=True,
        text=True
    )
    assert "error: the following arguments are required: --from" in result.stderr

def test_missing_to():
    result = subprocess.run(
        [sys.executable, 'cli.py', '--value', '100', '--from', 'celsius'],
        capture_output=True,
        text=True
    )
    assert "error: the following arguments are required: --to" in result.stderr

def test_invalid_unit():
    result = subprocess.run(
        [sys.executable, 'cli.py', '--value', '100', '--from', 'celsuis', '--to', 'fahrenheit'],
        capture_output=True,
        text=True
    )
    assert "invalid choice: 'celsuis'" in result.stderr
```