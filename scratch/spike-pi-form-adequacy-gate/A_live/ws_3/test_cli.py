import unittest
import sys
import io
from cli import main
from converters import celsius_to_fahrenheit, fahrenheit_to_celsius, celsius_to_kelvin

class TestCLI(unittest.TestCase):
    def test_celsius_to_fahrenheit(self):
        sys.argv = ['cli.py', 'celsius_to_fahrenheit', '25']
        captured = io.StringIO()
        sys.stdout = captured
        main()
        sys.stdout = sys.__stdout__
        self.assertEqual(captured.getvalue().strip(), '77.0')
        # Verify the actual converter function is used
        self.assertTrue(hasattr(main, 'converter_func'))  # Example placeholder for actual verification

    def test_fahrenheit_to_celsius(self):
        sys.argv = ['cli.py', 'fahrenheit_to_celsius', '77']
        captured = io.StringIO()
        sys.stdout = captured
        main()
        sys.stdout = sys.__stdout__
        self.assertEqual(captured.getvalue().strip(), '25.0')
        # Verify the actual converter function is used
        self.assertTrue(hasattr(main, 'converter_func'))  # Example placeholder for actual verification

    def test_celsius_to_kelvin(self):
        sys.argv = ['cli.py', 'celsius_to_kelvin', '100']
        captured = io.StringIO()
        sys.stdout = captured
        main()
        sys.stdout = sys.__stdout__
        self.assertEqual(captured.getvalue().strip(), '373.15')
        # Verify the actual converter function is used
        self.assertTrue(hasattr(main, 'converter_func'))  # Example placeholder for actual verification

    def test_invalid_command(self):
        sys.argv = ['cli.py', 'invalid_command', '42']
        captured = io.StringIO()
        sys.stdout = captured
        main()
        sys.stdout = sys.__stdout__
        self.assertIn('Invalid command', captured.getvalue())

    def test_missing_argument(self):
        sys.argv = ['cli.py', 'celsius_to_fahrenheit']
        captured = io.StringIO()
        sys.stdout = captured
        main()
        sys.stdout = sys.__stdout__
        self.assertIn('Missing argument', captured.getvalue())