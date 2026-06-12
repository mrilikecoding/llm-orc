import unittest
import cli
from converters import celsius_to_fahrenheit, fahrenheit_to_celsius, celsius_to_kelvin
from unittest.mock import patch, MagicMock

class TestCLI(unittest.TestCase):
    def test_celsius_to_fahrenheit(self):
        with patch('argparse.ArgumentParser.parse_args') as mock_parse_args:
            mock_parse_args.return_value = MagicMock(args={'celsius': 25})
            cli.main()
            celsius_to_fahrenheit.assert_called_once_with(25)

    def test_fahrenheit_to_celsius(self):
        with patch('argparse.ArgumentParser.parse_args') as mock_parse_args:
            mock_parse_args.return_value = MagicMock(args={'fahrenheit': 77})
            cli.main()
            fahrenheit_to_celsius.assert_called_once_with(77)

    def test_celsius_to_kelvin(self):
        with patch('argparse.ArgumentParser.parse_args') as mock_parse_args:
            mock_parse_args.return_value = MagicMock(args={'celsius': 0})
            cli.main()
            celsius_to_kelvin.assert_called_once_with(0)

if __name__ == '__main__':
    unittest.main()