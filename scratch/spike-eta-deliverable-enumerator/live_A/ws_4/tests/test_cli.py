

import pytest
import sys
from io import StringIO
from unittest.mock import patch

# Import from the actual module paths
sys.path.insert(0, '/Users/nathangreen/Development/eddi-lab/llm-orc/scratch/spike-eta-deliverable-enumerator/live_A/ws_4')

from temperature_conversion import celsius_to_fahrenheit, fahrenheit_to_celsius, celsius_to_kelvin


class TestCLIArgumentParsing:
    """Test that the CLI correctly parses command-line arguments."""
    
    def test_parse_celsius_to_fahrenheit(self, capsys):
        """Test parsing --c2f flag."""
        with patch(sys.argv, ['cli.py', '--c2f', '100']):
            from cli import main
            main()
            captured = capsys.readouterr()
            assert '212' in captured.out
    
    def test_parse_fahrenheit_to_celsius(self, capsys):
        """Test parsing --f2c flag."""
        with patch(sys.argv, ['cli.py', '--f2c', '32']):
            from cli import main
            main()
            captured = capsys.readouterr()
            assert '0' in captured.out
    
    def test_parse_celsius_to_kelvin(self, capsys):
        """Test parsing --c2k flag."""
        with patch(sys.argv, ['cli.py', '--c2k', '0']):
            from cli import main
            main()
            captured = capsys.readouterr()
            assert '273.15' in captured.out
    
    def test_missing_arguments(self, capsys):
        """Test that missing arguments produces an error."""
        with patch(sys.argv, ['cli.py']):
            from cli import main
            with pytest.raises(SystemExit):
                main()
    
    def test_invalid_flag(self, capsys):
        """Test that invalid flags produce an error."""
        with patch(sys.argv, ['cli.py', '--invalid', '100']):
            from cli import main
            with pytest.raises(SystemExit):
                main()


class TestConversionFunctions:
    """Test the three temperature conversion functions."""
    
    def test_celsius_to_fahrenheit_zero(self):
        """Test freezing point of water: 0C = 32F."""
        result = celsius_to_fahrenheit(0)
        assert result == 32
    
    def test_celsius_to_fahrenheit_100(self):
        """Test boiling point of water: 100C = 212F."""
        result = celsius_to_fahrenheit(100)
        assert result == 212
    
    def test_celsius_to_fahrenheit_negative_40(self):
        """Test -40 where C = F."""
        result = celsius_to_fahrenheit(-40)
        assert result == -40
    
    def test_fahrenheit_to_celsius_32(self):
        """Test freezing point of water: 32F = 0C."""
        result = fahrenheit_to_celsius(32)
        assert result == 0
    
    def test_fahrenheit_to_celsius_212(self):
        """Test boiling point of water: 212F = 100C."""
        result = fahrenheit_to_celsius(212)
        assert result == 100
    
    def test_fahrenheit_to_celsius_negative_40(self):
        """Test -40 where F = C."""
        result = fahrenheit_to_celsius(-40)
        assert result == -40
    
    def test_celsius_to_kelvin_zero(self):
        """Test freezing point of water: 0C = 273.15K."""
        result = celsius_to_kelvin(0)
        assert result == 273.15
    
    def test_celsius_to_kelvin_100(self):
        """Test boiling point of water: 100C = 373.15K."""
        result = celsius_to_kelvin(100)
        assert result == 373.15
    
    def test_celsius_to_kelvin_absolute_zero(self):
        """Test absolute zero: -273.15C = 0K."""
        result = celsius_to_kelvin(-273.15)
        assert result == 0


class TestErrorCases:
    """Test error handling for invalid inputs."""
    
    def test_non_numeric_value_celsius_to_fahrenheit(self):
        """Test that non-numeric input raises an error."""
        with pytest.raises((ValueError, TypeError)):
            celsius_to_fahrenheit('abc')
    
    def test_non_numeric_value_fahrenheit_to_celsius(self):
        """Test that non-numeric input raises an error."""
        with pytest.raises((ValueError, TypeError)):
            fahrenheit_to_celsius('xyz')
    
    def test_non_numeric_value_celsius_to_kelvin(self):
        """Test that non-numeric input raises an error."""
        with pytest.raises((ValueError, TypeError)):
            celsius_to_kelvin('not a number')
    
    def test_invalid_unit_handling(self, capsys):
        """Test that invalid unit flags produce an error."""
        with patch(sys.argv, ['cli.py', '--c2x', '100']):
            from cli import main
            with pytest.raises(SystemExit):
                main()


class TestOutputFormatting:
    """Test that output is properly formatted."""
    
    def test_output_format_with_precision(self, capsys):
        """Test output includes decimal precision."""
        with patch(sys.argv, ['cli.py', '--c2k', '37']):
            from cli import main
            main()
            captured = capsys.readouterr()
            assert '310.15' in captured.out
    
    def test_output_includes_unit_label(self, capsys):
        """Test output includes the target unit."""
        with patch(sys.argv, ['cli.py', '--c2f', '0']):
            from cli import main
            main()
            captured = capsys.readouterr()
            assert 'F' in captured.out or 'Fahrenheit' in captured.out
    
    def test_negative_temperature_output(self, capsys):
        """Test output for negative temperatures."""
        with patch(sys.argv, ['cli.py', '--c2f', '-10']):
            from cli import main
            main()
            captured = capsys.readouterr()
            assert '14' in captured.out
    
    def test_decimal_input_formatting(self, capsys):
        """Test output formatting with decimal inputs."""
        with patch(sys.argv, ['cli.py', '--c2f', '36.6']):
            from cli import main
            main()
            captured = capsys.readouterr()
            assert '97.88' in captured.out or '97.9' in captured.out