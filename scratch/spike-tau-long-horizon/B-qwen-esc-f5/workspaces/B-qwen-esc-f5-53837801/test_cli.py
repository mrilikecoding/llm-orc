import unittest
from unittest import mock
import cli
import converters

class TestCli(unittest.TestCase):
    def test_parse_args(self):
        args = cli.parse_args(['--from', 'celsius', '--to', 'fahrenheit', '--value', '0'])
        self.assertEqual(args.from_unit, 'celsius')
        self.assertEqual(args.to_unit, 'fahrenheit')
        self.assertEqual(args.value, 0.0)
        
        # Test invalid unit
        with self.assertRaises(SystemExit):
            cli.parse_args(['--from', 'invalid', '--to', 'fahrenheit', '--value', '0'])

    def test_main_conversion_celsius_to_fahrenheit(self):
        mock_args = mock.MagicMock()
        mock_args.from_unit = 'celsius'
        mock_args.to_unit = 'fahrenheit'
        mock_args.value = 0.0
        with mock.patch('cli.parse_args', return_value=mock_args):
            with mock.patch('converters.celsius_to_fahrenheit') as mock_func:
                cli.main()
                mock_func.assert_called_once_with(0.0)

    def test_main_conversion_fahrenheit_to_celsius(self):
        mock_args = mock.MagicMock()
        mock_args.from_unit = 'fahrenheit'
        mock_args.to, 'celsius'
        mock_args.value = 32.0
        with mock.patch('cli.parse_args', return_value=mock_args):
            with mock.patch('converters.fahrenheit_to_celsius') as mock_func:
                cli.main()
                mock_func.assert_called_once_with(32.0)

    def test_main_conversion_celsius_to_kelvin(self):
        mock_args = mock.MagicMock()
        mock_args.from_unit = 'celsius'
        mock_args.to_unit = 'kelvin'
        mock_args.value = 0.0
        with mock.patch('cli.parse_args', return_value=mock_args):
            with mock.patch('converters.celsius_to_kelvin') as mock_func:
                cli.main()
                mock_func.assert_called_once_with(0.0)

    def test_main_invalid_unit(self):
        mock_args = mock.MagicMock()
        mock_args.from_unit = 'invalid'
        mock_args.to_unit = 'fahrenheit'
        mock_args.value = 0.0
        with mock.patch('cli.parse_args', return_value=mock_args):
            with self.assertRaises(SystemExit):
                cli.main()