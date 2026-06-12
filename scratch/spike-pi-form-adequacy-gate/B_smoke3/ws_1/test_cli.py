import pytest
import sys
from cli import main

def test_argument_parsing_celsius_to_fahrenheit(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['temperature-converter', 'celsius', 'fahrenheit', '25'])
    main()
    captured = capsys.readouterr()
    assert captured.out == "Fahrenheit: 77.0\n"

def test_argument_parsing_fahrenheit_to_celsius(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['temperature-converter', 'fahrenheit', 'celsius', '77'])
    main()
    captured = capsys.readouterr()
    assert captured.out == "Celsius: 25.0\n"

def test_argument_parsing_celsius_to_kelvin(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['temperature-converter', 'celsius', 'kelvin', '300'])
    main()
    captured = capsys.readouterr()
    assert captured.out == "Kelvin: 573.15\n"

def test_invalid_temperature_input(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['temperature-converter', 'celsius', 'fahrenheit', 'abc'])
    main()
    captured = capsys.readouterr()
    assert "Invalid temperature value" in captured.err

def test_invalid_conversion_pair(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['temperature-converter', 'fahrenheit', 'kelvin', '32'])
    main()
    captured = capsys.readouterr()
    assert "Unsupported conversion" in captured.err

def test_missing_arguments(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['temperature-converter', 'celsius', 'fahrenheit'])
    main()
    captured = capsys.readouterr()
    assert "Missing temperature value" in captured.err