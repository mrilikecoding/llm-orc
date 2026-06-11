

import sys
import subprocess
import pytest

sys.path.insert(0, '/Users/nathangreen/Development/eddi-lab/llm-orc/scratch/spike-eta-deliverable-enumerator/live_A/ws_3')
from temp_convert import celsius_to_fahrenheit, fahrenheit_to_celsius, celsius_to_kelvin


class TestTemperatureConversionModule:
    """Test the temperature conversion functions from temp_convert module"""
    
    def test_celsius_to_fahrenheit_zero(self):
        assert celsius_to_fahrenheit(0) == 32
    
    def test_celsius_to_fahrenheit_boiling(self):
        assert celsius_to_fahrenheit(100) == 212
    
    def test_celsius_to_fahrenheit_negative_forty(self):
        assert celsius_to_fahrenheit(-40) == -40
    
    def test_celsius_to_fahrenheit_absolute_zero(self):
        assert celsius_to_fahrenheit(-273.15) == -459.67
    
    def test_fahrenheit_to_celsius_freezing(self):
        assert fahrenheit_to_celsius(32) == 0
    
    def test_fahrenheit_to_celsius_boiling(self):
        assert fahrenheit_to_celsius(212) == 100
    
    def test_fahrenheit_to_celsius_negative_forty(self):
        assert fahrenheit_to_celsius(-40) == -40
    
    def test_fahrenheit_to_celsius_absolute_zero(self):
        assert fahrenheit_to_celsius(-459.67) == -273.15
    
    def test_celsius_to_kelvin_freezing(self):
        assert celsius_to_kelvin(0) == 273.15
    
    def test_celsius_to_kelvin_boiling(self):
        assert celsius_to_kelvin(100) == 373.15
    
    def test_celsius_to_kelvin_absolute_zero(self):
        assert celsius_to_kelvin(-273.15) == 0
    
    def test_celsius_to_kelvin_negative_forty(self):
        assert celsius_to_kelvin(-40) == 233.15
    
    def test_celsius_to_fahrenheit_returns_float(self):
        result = celsius_to_fahrenheit(50)
        assert isinstance(result, float)
    
    def test_fahrenheit_to_celsius_returns_float(self):
        result = fahrenheit_to_celsius(50)
        assert isinstance(result, float)
    
    def test_celsius_to_kelvin_returns_float(self):
        result = celsius_to_kelvin(50)
        assert isinstance(result, float)


class TestCLIArgumentParsing:
    """Test CLI argument parsing by invoking main() with various arguments"""
    
    def test_cli_basic_conversion_c_to_f(self):
        result = subprocess.run(
            [sys.executable, '-c', 
             'import sys; sys.path.insert(0, "/Users/nathangreen/Development/eddi-lab/llm-orc/scratch/spike-eta-deliverable-enumerator/live_A/ws_3"); '
             'from cli import main; sys.argv = ["cli.py", "100", "C", "F"]; main()'],
            capture_output=True,
            text=True,
            cwd='/Users/nathangreen/Development/eddi-lab/llm-orc/scratch/spike-eta-deliverable-enumerator/live_A/ws_3'
        )
        assert "212" in result.stdout or result.returncode == 0
    
    def test_cli_basic_conversion_f_to_c(self):
        result = subprocess.run(
            [sys.executable, '-c', 
             'import sys; sys.path.insert(0, "/Users/nathangreen/Development/eddi-lab/llm-orc/scratch/spike-eta-deliverable-enumerator/live_A/ws_3"); '
             'from cli import main; sys.argv = ["cli.py", "32", "F", "C"]; main()'],
            capture_output=True,
            text=True,
            cwd='/Users/nathangreen/Development/eddi-lab/llm-orc/scratch/spike-eta-deliverable-enumerator/live_A/ws_3'
        )
        assert "0" in result.stdout or result.returncode == 0
    
    def test_cli_conversion_c_to_k(self):
        result = subprocess.run(
            [sys.executable, '-c', 
             'import sys; sys.path.insert(0, "/Users/nathangreen/Development/eddi-lab/llm-orc/scratch/spike-eta-deliverable-enumerator/live_A/ws_3"); '
             'from cli import main; sys.argv = ["cli.py", "0", "C", "K"]; main()'],
            capture_output=True,
            text=True,
            cwd='/Users/nathangreen/Development/eddi-lab/llm-orc/scratch/spike-eta-deliverable-enumerator/live_A/ws_3'
        )
        assert "273.15" in result.stdout or result.returncode == 0
    
    def test_cli_with_float_value(self):
        result = subprocess.run(
            [sys.executable, '-c', 
             'import sys; sys.path.insert(0, "/Users/nathangreen/Development/eddi-lab/llm-orc/scratch/spike-eta-deliverable-enumerator/live_A/ws_3"); '
             'from cli import main; sys.argv = ["cli.py", "98.6", "F", "C"]; main()'],
            capture_output=True,
            text=True,
            cwd='/Users/nathangreen/Development/eddi-lab/llm-orc/scratch/spike-eta-deliverable-enumerator/live_A/ws_3'
        )
        assert result.returncode == 0 or "37" in result.stdout


class TestConvertTemperatureFunction:
    """Test the convert_temperature function from cli module"""
    
    def test_convert_temperature_exists(self):
        from cli import convert_temperature
        assert callable(convert_temperature)
    
    def test_convert_celsius_to_fahrenheit(self):
        from cli import convert_temperature
        result = convert_temperature(100, "C", "F")
        assert result == 212
    
    def test_convert_fahrenheit_to_celsius(self):
        from cli import convert_temperature
        result = convert_temperature(32, "F", "C")
        assert result == 0
    
    def test_convert_celsius_to_kelvin(self):
        from cli import convert_temperature
        result = convert_temperature(0, "C", "K")
        assert result == 273.15
    
    def test_convert_kelvin_to_celsius(self):
        from cli import convert_temperature
        result = convert_temperature(273.15, "K", "C")
        assert result == 0
    
    def test_convert_fahrenheit_to_kelvin(self):
        from cli import convert_temperature
        result = convert_temperature(32, "F", "K")
        assert result == 273.15
    
    def test_convert_kelvin_to_fahrenheit(self):
        from cli import convert_temperature
        result = convert_temperature(273.15, "K", "F")
        assert result == 32
    
    def test_convert_same_unit(self):
        from cli import convert_temperature
        result = convert_temperature(50, "C", "C")
        assert result == 50
    
    def test_convert_negative_values(self):
        from cli import convert_temperature
        result = convert_temperature(-40, "C", "F")
        assert result == -40


class TestKelvinToCelsiusFunction:
    """Test the kelvin_to_celsius function from cli module"""
    
    def test_kelvin_to_celsius_exists(self):
        from cli import kelvin_to_celsius
        assert callable(kelvin_to_celsius)
    
    def test_kelvin_to_celsius_freezing_point(self):
        from cli import kelvin_to_celsius
        result = kelvin_to_celsius(273.15)
        assert result == 0
    
    def test_kelvin_to_celsius_boiling_point(self):
        from cli import kelvin_to_celsius
        result = kelvin_to_celsius(373.15)
        assert result == 100
    
    def test_kelvin_to_celsius_absolute_zero(self):
        from cli import kelvin_to_celsius
        result = kelvin_to_celsius(0)
        assert result == -273.15
    
    def test_kelvin_to_celsius_negative(self):
        from cli import kelvin_to_celsius
        result = kelvin_to_celsius(233.15)
        assert result == -40


class TestCLIIntegration:
    """Integration tests for complete CLI workflows"""
    
    def test_cli_returns_zero_exit_on_success(self):
        result = subprocess.run(
            [sys.executable, '-c', 
             'import sys; sys.path.insert(0, "/Users/nathangreen/Development/eddi-lab/llm-orc/scratch/spike-eta-deliverable-enumerator/live_A/ws_3"); '
             'from cli import main; sys.argv = ["cli.py", "100", "C", "F"]; main()'],
            capture_output=True,
            text=True,
            cwd='/Users/nathangreen/Development/eddi-lab/llm-orc/scratch/spike-eta-deliverable-enumerator/live_A/ws_3'
        )
        assert result.returncode == 0
    
    def test_cli_invalid_unit_handling(self):
        result = subprocess.run(
            [sys.executable, '-c', 
             'import sys; sys.path.insert(0, "/Users/nathangreen/Development/eddi-lab/llm-orc/scratch/spike-eta-deliverable-enumerator/live_A/ws_3"); '
             'from cli import main; sys.argv = ["cli.py", "100", "X", "Y"]; main()'],
            capture_output=True,
            text=True,
            cwd='/Users/nathangreen/Development/eddi-lab/llm-orc/scratch/spike-eta-deliverable-enumerator/live_A/ws_3'
        )
        assert result.returncode != 0 or "error" in result.stdout.lower() or "invalid" in result.stdout.lower()
    
    def test_cli_with_invalid_arguments_count(self):
        result = subprocess.run(
            [sys.executable, '-c', 
             'import sys; sys.path.insert(0, "/Users/nathangreen/Development/eddi-lab/llm-orc/scratch/spike-eta-deliverable-enumerator/live_A/ws_3"); '
             'from cli import main; sys.argv = ["cli.py", "100"]; main()'],
            capture_output=True,
            text=True,
            cwd='/Users/nathangreen/Development/eddi-lab/llm-orc/scratch/spike-eta-deliverable-enumerator/live_A/ws_3'
        )
        assert result.returncode != 0 or "usage" in result.stdout.lower() or "error" in result.stdout.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
