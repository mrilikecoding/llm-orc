import subprocess
import unittest

class TestCLI(unittest.TestCase):
    def test_celsius_to_fahrenheit(self):
        result = subprocess.run(['python', 'cli.py', '25', 'celsius', 'fahrenheit'], capture_output=True, text=True)
        self.assertEqual(result.stdout.strip(), '77.0')

    def test_fahrenheit_to_celsius(self):
        result = subprocess.run(['python', 'cli.py', '32', 'fahrenheit', 'celsius'], capture_output=True, text=True)
        self.assertEqual(result.stdout.strip(), '0.0')

    def test_celsius_to_kelvin(self):
        result = subprocess.run(['python', 'cli.py', '0', 'celsius', 'kelvin'], capture_output=True, text=True)
        self.assertEqual(result.stdout.strip(), '273.15')

    def test_unknown_unit_from(self):
        result = subprocess.run(['python', 'cli.py', '100', 'kelvin', 'celsius'], capture_output=True, text=True)
        self.assertIn('error', result.stderr.lower())

    def test_unknown_unit_to(self):
        result = subprocess.run(['python', 'cli.py', '100', 'celsius', 'rankine'], capture_output=True, text=True)
        self.assertIn('error', result.stderr.lower())

    def test_missing_arguments(self):
        result = subprocess.run(['python', 'cli.py', '100', 'celsius'], capture_output=True, text=True)
        self.assertIn('error', result.stderr.lower())
        result = subprocess.run(['python', 'cli.py'], capture_output=True, text=True)
        self.assertIn('error', result.stderr.lower())

    def test_output_formatting(self):
        result = subprocess.run(['python', 'cli.py', '25', 'celsius', 'fahrenheit'], capture_output=True, text=True)
        self.assertEqual(result.stdout.strip(), '77.0')