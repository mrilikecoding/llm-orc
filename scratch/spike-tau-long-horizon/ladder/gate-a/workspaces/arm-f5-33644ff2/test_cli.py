import unittest
import sys
from io import StringIO
import cli
import converters

class TestCLI(unittest.TestCase):
    def test_valid_conversion_celsius_to_fahrenheit(self):
        mock_converter = unittest.mock.MagicMock(return_value=32.0)
        with unittest.mock.patch('converters.celsius_to_fahrenheit', mock_converter):
            sys.argv = ['script.py', '0', 'Celsius', 'Fahrenheit']
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            cli.main()
            sys.stdout = old_stdout
            mock_converter.assert_called_once_with(0)
            self.assertEqual(sys.stdout.getvalue().strip(), '0 Celsius is 32.0 Fahrenheit')

    def test_valid_conversion_fahrenheit_to_celsius(self):
        mock_converter = unittest.mock.MagicMock(return_value=0.0)
        with unittest.mock.patch('converters.fahrenheit_to_celsius', mock_converter):
            sys.argv = ['script.py', '32', 'Fahrenheit', 'Celsius']
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            cli.main()
            sys.stdout = old_stdout
            mock_converter.assert_called_once_with(32)
            self.assertEqual(sys.stdout.getvalue().strip(), '32 Fahrenheit is 0.0 Celsius')

    def test_valid_conversion_celsius_to_kelvin(self):
        mock_converter = unittest.mock.MagicMock(return_value=273.15)
        with unittest.mock.patch('converters.celsius_to_kelvin', mock_converter):
            sys.argv = ['script.py', '0', 'Celsius', 'Kelvin']
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            cli.main()
            sys.stdout = old_stdout
            mock_converter.assert_called_once_with(0)
            self.assertEqual(sys.stdout.getvalue().strip(), '0 Celsius is 273.15 Kelvin')

    def test_invalid_conversion(self):
        with unittest.mock.patch('sys.exit') as mock_exit:
            sys.argv = ['script.py', '0', 'Invalid', 'Celsius']
            cli.main()
            mock_exit.assert_called_once_with(1)