

import unittest
import subprocess
import sys


class TestCLIConvertFunction(unittest.TestCase):
    """Tests for the convert function used by the CLI."""
    
    def test_celsius_to_fahrenheit_100(self):
        from temperature import celsius_to_fahrenheit
        result = celsius_to_fahrenheit(100)
        self.assertAlmostEqual(result, 212.0, places=1)
    
    def test_fahrenheit_to_celsius_212(self):
        from temperature import fahrenheit_to_celsius
        result = fahrenheit_to_celsius(212)
        self.assertAlmostEqual(result, 100.0, places=1)
    
    def test_celsius_to_kelvin_absolute_zero(self):
        from temperature import celsius_to_kelvin
        result = celsius_to_kelvin(-273)
        self.assertAlmostEqual(result, 0.0, places=1)


class TestCLIIntegration(unittest.TestCase):
    """Integration tests for the CLI tool via subprocess."""
    
    def test_convert_100_celsius_to_fahrenheit(self):
        result = subprocess.run(
            [sys.executable, '-c', 
             'import sys; sys.argv = ["", "100", "celsius", "fahrenheit"]; '
             'from index import main; main()'],
            capture_output=True,
            text=True,
            cwd='/Users/nathangreen/Development/eddi-lab/llm-orc/scratch/spike-eta-deliverable-enumerator/live_A/ws_2/cli'
        )
        self.assertIn('212', result.stdout)
    
    def test_convert_212_fahrenheit_to_celsius(self):
        result = subprocess.run(
            [sys.executable, '-c', 
             'import sys; sys.argv = ["", "212", "fahrenheit", "celsius"]; '
             'from index import main; main()'],
            capture_output=True,
            text=True,
            cwd='/Users/nathangreen/Development/eddi-lab/llm-orc/scratch/spike-eta-deliverable-enumerator/live_A/ws_2/cli'
        )
        self.assertIn('100', result.stdout)
    
    def test_convert_negative_273_celsius_to_kelvin(self):
        result = subprocess.run(
            [sys.executable, '-c', 
             'import sys; sys.argv = ["", "-273", "celsius", "kelvin"]; '
             'from index import main; main()'],
            capture_output=True,
            text=True,
            cwd='/Users/nathangreen/Development/eddi-lab/llm-orc/scratch/spike-eta-deliverable-enumerator/live_A/ws_2/cli'
        )
        self.assertIn('-0.15', result.stdout)


class TestConvertFunctionDirectly(unittest.TestCase):
    """Test the convert function by importing it directly."""
    
    def setUp(self):
        import sys
        sys.path.insert(0, '/Users/nathangreen/Development/eddi-lab/llm-orc/scratch/spike-eta-deliverable-enumerator/live_A/ws_2/cli')
        from index import convert
        self.convert = convert
    
    def test_convert_100_celsius_to_fahrenheit(self):
        result = self.convert(100, 'celsius', 'fahrenheit')
        self.assertAlmostEqual(result, 212.0, places=1)
    
    def test_convert_212_fahrenheit_to_celsius(self):
        result = self.convert(212, 'fahrenheit', 'celsius')
        self.assertAlmostEqual(result, 100.0, places=1)
    
    def test_convert_negative_273_celsius_to_kelvin(self):
        result = self.convert(-273, 'celsius', 'kelvin')
        self.assertAlmostEqual(result, 0.15, places=1)
    
    def test_convert_same_unit(self):
        result = self.convert(50, 'celsius', 'celsius')
        self.assertEqual(result, 50)
    
    def test_convert_fahrenheit_to_kelvin(self):
        result = self.convert(32, 'fahrenheit', 'kelvin')
        self.assertAlmostEqual(result, 273.15, places=1)
    
    def test_convert_kelvin_to_celsius(self):
        result = self.convert(273.15, 'kelvin', 'celsius')
        self.assertAlmostEqual(result, 0.0, places=1)
    
    def test_convert_kelvin_to_fahrenheit(self):
        result = self.convert(273.15, 'kelvin', 'fahrenheit')
        self.assertAlmostEqual(result, 32.0, places=1)
    
    def test_convert_unsupported(self):
        with self.assertRaises(ValueError):
            self.convert(100, 'celsius', 'rankine')


if __name__ == '__main__':
    unittest.main()