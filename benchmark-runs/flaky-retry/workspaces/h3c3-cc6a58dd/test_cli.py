import unittest
import subprocess
from converters import celsius_to_fahrenheit, celsius_to_kelvin

class TestCli(unittest.TestCase):
    def test_valid_conversion_to_fahrenheit(self):
        result = subprocess.run(['python', 'cli.py', '100', '--to-fahrenheit'], capture_output=True, text=True)
        self.assertEqual(result.stdout.strip(), str(celsius_to_fahrenheit(100)))
    
    def test_valid_conversion_to_kelvin(self):
        result = subprocess.run(['python', 'cli.py', '0', '--to-kelvin'], capture_output=True, text=True)
        self.assertEqual(result.stdout.strip(), str(celsius_to_kelvin(0)))
    
    def test_invalid_input(self):
        result = subprocess.run(['python', 'cli.py', 'abc', '--to-fahrenheit'], capture_output=True, text=True)
        self.assertIn('invalid literal for int()', result.stderr)
    
    def test_missing_value(self):
        result = subprocess.run(['python', 'cli.py', '--to-fahrenheit'], capture_output=True, text=True)
        self.assertIn('argument --to-fahrenheit: expected one argument', result.stderr)
    
    def test_both_flags(self):
        result = subprocess.run(['python', 'cli.py', '100', '--to-fahrenheit', '--to-kelvin'], capture_output=True, text=True)
        self.assertIn('conflicting arguments', result.stderr)