import pytest
from click.testing import CliRunner
from cli import main

def test_cli_valid_conversion():
    runner = CliRunner()
    result = runner.invoke(main, ["--from", "celsius", "--to", "fahrenheit", "0"])
    assert result.exit_code == 0
    assert result.output == "32.0\n"

def test_cli_fahrenheit_to_celsius():
    runner = CliRunner()
    result = runner.invoke(main, ["--from", "fahrenheit", "--to", "celsius", "212"])
    assert result.exit_code == 0
    assert result.output == "100.0\n"

def test_cli_celsius_to_kelvin():
    runner = CliRunner()
    result = runner.invoke(main, ["--from", "celsius", "--to", "kelvin", "273.15"])
    assert result.exit_code == 0
    assert result.output == "546.3\n"

def test_cli_invalid_input():
    runner = CliRunner()
    result = runner.invoke(main, ["--from", "celsius", "--to", "fahrenheit", "abc"])
    assert result.exit_code == 2
    assert "Invalid value" in result.output

def test_cli_argument_parsing():
    runner = CliRunner()
    result = runner.invoke(main, ["--from", "invalid", "--to", "fahrenheit", "0"])
    assert result.exit_code == 2
    assert "Invalid value for --from" in result.output

def test_cli_missing_arguments():
    runner = CliRunner()
    result = runner.invoke(main, ["--from", "celsius", "--to", "fahrenheit"])
    assert result.exit_code == 2
    assert "Missing value" in result.output