import unittest
import unittest.mock
import sys
import io
import cli
from converters import celsius_to_fahrenheit, fahrenheit_to_celsius, celsius_to_kelvin

class TestCli(unittest.TestCase):
    def test_celsius_to_fahrenheit(self):
        with unittest.mock.patch('sys.stdout', new=io.StringIO()) as fake_out:
            with unittest.mock.patch('argparse.ArgumentParser') as mock_parser:
                mock_args = unittest.mock.Mock()
                mock_args.from_unit = 'celsius'
                mock_args.to_unit = 'fahrenheit'
                mock_args.value = 25
                mock_parser.return_value.parse_args.return_value = mock_args
                with unittest.mock.patch('converters.celsius_to_fahrenheit') as mock_convert:
                    cli.main()
                    mock_convert.assert_called_once_with(25)
                    self.assertEqual(fake_out.getvalue().strip(), '25°C is 77.0°F')

    def test_fahrenheit_to_celsius(self):
        with unittest.mock.patch('sys.stdout', new=io.StringIO()) as fake_out:
            with unittest.mock.patch('argparse.ArgumentParser') as mock_parser:
                mock_args = unittest.mock.Mock()
                mock_args.from_unit = 'fahrenheit'
                mock_args.to_unit = 'celsius'
                mock_args.value = 77
                mock_parser.return_value.parse_args.return_value = mock_args
                with unittest.mock.patch('converters.fahrenheit_to_celsius') as mock_convert:
                    cli.main()
                    mock_convert.assert_called_once_with(77)
                    self.assertEqual(fake_out.getvalue().strip(), '77°F is 25.0°C')

    def test_celsius_to_kelvin(self):
        with unittest.mock.patch('sys.stdout', new=io.StringIO()) as fake_out:
            with unittest.mock.patch('argparse.ArgumentParser') as mock_parser:
                mock_args = unittest.mock.Mock()
                mock_args.from_unit = 'celsius'
                mock_args.to_unit = 'kelvin'
                mock_args.value = 25
                mock_parser.return_value.parse_args.return_value = mock_args
                with unittest.mock.patch('converters.celsius_to_kelvin') as mock_convert:
                    cli.main()
                    mock_convert.assert_called_once_with(25)
                    self.assertEqual(fake_out.getvalue().strip(), '25°C is 298.15 K')

    def test_invalid_unit(self):
        with unittest.mock.patch('sys.stdout', new=io.StringIO()) as fake_out:
            with unittest.mock.patch('argparse.ArgumentParser') as mock_parser:
                mock_args = unittest.mock.Mock()
                mock_args.from_unit = 'invalid'
                mock_args.to_unit = 'fahrenheit'
                mock_args.value = 25
                mock_parser.return_value.parse_args.return_value = mock_args
                with unittest.mock.patch('converters.celsius_to_fahrenheit') as mock_c_to_f:
                    with unittest.mock.patch('converters.fahrenheit_to_celsius') as mock_f_to_c:
                        with unittest.mock.patch('converters.celsius_to_kelvin') as mock_c_to_k:
                            cli.main()
                            mock_c_to_f.assert_not_called()
                            mock_f_to_c.assert_not_called()
                            mock_c_to_k.assert_not_called()
                            self.assertIn('Invalid unit', fake_out.getvalue())

    def test_missing_arguments(self):
        with self.assertRaises(SystemExit):
            with unittest.mock.patch('argparse.ArgumentParser') as mock_parser:
                mock_parser.return_value.parse_args.side_effect = argparse.ArgumentError("Missing required argument")
                cli.main()