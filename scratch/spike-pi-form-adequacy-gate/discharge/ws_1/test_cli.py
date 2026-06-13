import cli
import unittest
from unittest import mock

class TestCLI(unittest.TestCase):
    def test_convert_arguments(self):
        with mock.patch('sys.argv', ['cli.py', '--from-unit', 'Celsius', '--to-unit', 'Fahrenheit', '25']):
            cli.main()
            cli.convert.assert_called_with(25, 'Celsius', 'Fahrenheit')

    def test_conversion_logic(self):
        with mock.patch('converters.celsius_to_fahrenheit') as mock_celsius_to_fahrenheit:
            result = cli.convert(25, 'Celsius', 'Fahrenheit')
            mock_celsius_to_fahrenheit.assert_called_with(25)
            self.assertAlmostEqual(result, 77.0, delta=0.01)

    def test_invalid_unit(self):
        with self.assertRaises(ValueError):
            cli.convert(25, 'Invalid', 'Fahrenheit')

if __name__ == '__main__':
    unittest.main()