import unittest
from cli import main
import sys
from unittest.mock import patch, MagicMock, Mock

class TestCLI(unittest.TestCase):
    def setUp(self):
        self.original_argv = sys.argv[:]
        sys.argv = ['cli.py']

    def tearDown(self):
        sys.argv = self.original_argv

    def test_main_celsius_to_fahrenheit(self):
        with patch('sys.argv', ['cli.py', '--celsius', '25', '--to', 'fahrenheit']):
            with patch('builtins.print') as mock_print:
                with patch('converters.celsius_to_fahrenheit') as mock_converter:
                    main()
                    mock_converter.assert_called_with(25)
                    mock_print.assert_called_with("25°C is 77.0°F")

    def test_main_fahrenheit_to_celsius(self):
        with patch('sys.argv', ['cli.py', '--fahrenheit', '77', '--to', 'celsius']):
            with patch('builtins.print') as mock_print:
                with patch('converters.fahrenheit_to_celsius') as mock_converter:
                    main()
                    mock_converter.assert_called_with(77)
                    mock_print.assert_called_with("77°F is 25.0°C")

    def test_main_celsius_to_kelvin(self):
        with patch('sys.argv', ['cli.py', '--celsius', '100', '--to', 'kelvin']):
            with patch('builtins.print') as mock_print:
                with patch('converters.celsius_to_kelvin') as mock_converter:
                    main()
                    mock_converter.assert_called_with(100)
                    mock_print.assert_called_with("100°C is 373.15K")

    def test_invalid_input(self):
        with patch('sys.argv', ['cli.py', '--celsius', 'invalid', '--to', 'fahrenheit']):
            with patch('builtins.print') as mock_print:
                with self.assertRaises(ValueError):
                    main()
                mock_print.assert_called_with("Error: Invalid value 'invalid' for --celsius")

if __name__ == '__main__':
    unittest.main()