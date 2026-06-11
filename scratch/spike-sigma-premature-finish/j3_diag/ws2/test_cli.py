import sys
import pytest
from io import StringIO
from cli import main

def test_celsius_to_fahrenheit(capsys):
    sys.argv = ['cli.py', '100', '--from', 'celsius', '--to', 'fahrenheit']
    main()
    captured = capsys.readouterr()
    assert captured.out == "100 celsius is equal to 212.0 fahrenheit\n"

def test_celsius_to_kelvin(capsys):
    sys.argv = ['cli.py', '0', '--from', 'celsius', '--to', 'kelvin']
    main()
    captured = capsys.readouterr()
    assert captured.out == "0 celsius is equal to 273.15 kelvin\n"

def test_fahrenheit_to_celsius(capsys):
    sys.argv = ['cli.py', '32', '--from', 'fahrenheit', '--to', 'celsius']
    main()
    captured = capsys.readouterr()
    assert captured.out == "32 fahrenheit is equal to 0.0 celsius\n"

def test_invalid_from_unit(capsys):
    sys.argv = ['cli.py', '100', '--from', 'kelvin', '--to', 'celsius']
    with pytest.raises(SystemExit):
        main()
    captured = capsys.readouterr()
    assert "error: invalid choice: 'kelvin' (choose from 'celsius', 'fahrenheit')" in captured.err

def test_invalid_to_unit(capsys):
    sys.argv = ['cli.py', '100', '--from', 'celsius', '--to', 'invalidunit']
    with pytest.raises(SystemExit):
        main()
    captured = capsys.readouterr()
    assert "error: invalid choice: 'invalidunit' (choose from 'celsius', 'fahrenheit')" in captured.err

def test_unsupported_conversion():
    sys.argv = ['cli.py', '32', '--from', 'fahrenheit', '--to', 'kelvin']
    with pytest.raises(ValueError) as exc_info:
        main()
    assert "Conversion from fahrenheit to kelvin is not supported." in str(exc_info.value)