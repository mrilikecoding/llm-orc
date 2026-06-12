import unittest
import argparse
import io
import sys
import cli

class TestCLI(unittest.TestCase):
    def test_argument_parsing_celsius_to_fahrenheit(self):
        args = argparse.Namespace(
            from_celsius=True,
            to='fahrenheit',
            temperature=25.0
        )
        self.assertEqual(cli.main(args), "25.0 celsius is 77.0 fahrenheit")

    def test_argument_parsing_fahrenheit_to_celsius(self):
        args = argparse.Namespace(
            from_fahrenheit=True,
            to='celsius',
            temperature=77.0
        )
        self.assertEqual(cli.main(args), "77.0 fahrenheit is 25.0 celsius")

    def test_argument_parsing_celsius_to_kelvin(self):
        args = argparse.Namespace(
            from_celsius=True,
            to='kelvin',
            temperature=0.0
        )
        self.assertEqual(cli.main(args), "0.0 celsius is 273.15 kelvin")

    def test_argument_parsing_fahrenheit_to_kelvin(self):
        args = argparse.Namespace(
            from_fahrenheit=True,
            to='kelvin',
            temperature=32.0
        )
        self.assertEqual(cli.main(args), "32.0 fahrenheit is 273.15 kelvin")

    def test_argument_parsing_kelvin_to_celsius(self):
        args = argparse.Namespace(
            from_kelvin=True,
            to='celsius',
            temperature=273.15
        )
        self.assertEqual(cli.main(args), "273.15 kelvin is 0.0 celsius")

    def test_argument_parsing_kelvin_to_fahrenheit(self):
        args = argparse.Namespace(
            from_kelvin=True,
            to='fahrenheit',
            temperature=273.15
        )
        self.assertEqual(cli.main(args), "273.15 kelvin is 32.0 fahrenheit")

    def test_invalid_conversion_fahrenheit_to_kelvin(self):
        parser = argparse.ArgumentParser(description='Convert temperature between units.')
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument('--from-celsius', action='store_true')
        group.add_argument('--from-fahrenheit', action='store_true')
        group.add_argument('--from-kelvin', action='store_true')
        parser.add_argument('--to', required=True, choices=['celsius', 'fahrenheit', 'kelvin'])
        parser.add_argument('temperature', type=float, help='Temperature value')
        with self.assertRaises(argparse.ArgumentError):
            parser.parse_args(['--from-fahrenheit', '--to', 'kelvin', '32'])

    def test_invalid_conversion_kelvin_to_fahrenheit(self):
        parser = argparse.ArgumentParser(description='Convert temperature between units.')
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument('--from-celsius', action='store_true')
        group.add_argument('--from-fahrenheit', action='store_true')
        group.add_argument('--from-kelvin', action='store_true')
        parser.add_argument('--to', required=True, choices=['celsius', 'fahrenheit', 'kelvin'])
        parser.add_argument('temperature', type=float, help='Temperature value')
        with self.assertRaises(argparse.ArgumentError):
            parser.parse_args(['--from-kelvin', '--to', 'fahrenheit', '273.15'])

    def test_invalid_conversion_same_unit(self):
        parser = argparse.ArgumentParser(description='Convert temperature between units.')
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument('--from-celsius', action='store_true')
        group.add_argument('--from-fahrenheit', action='store_true')
        group.add_argument('--from-kelvin', action='store_true')
        parser.add_argument('--to', required=True, choices=['celsius', 'fahrenheit', 'kelvin'])
        parser.add_argument('temperature', type=float, help='Temperature value')
        with self.assertRaises(argparse.ArgumentError):
            parser.parse_args(['--from-celsius', '--to', 'celsius', '25'])

    def test_output_celsius_to_fahrenheit(self):
        captured = io.StringIO()
        sys.stdout = captured
        args = argparse.Namespace(
            from_celsius=True,
            to='fahrenheit',
            temperature=25.0
        )
        cli.main(args)
        sys.stdout = sys.__stdout__
        self.assertEqual(captured.getvalue().strip(), "25.0 celsius is 77.0 fahrenheit")

    def test_output_fahrenheit_to_celsius(self):
        captured = io.StringIO()
        sys.stdout = captured
        args = argparse.Namespace(
            from_fahrenheit=True,
            to='celsius',
            temperature=77.0
        )
        cli.main(args)
        sys.stdout = sys.__stdout__
        self.assertEqual(captured.getvalue().strip(), "77.0 fahrenheit is 25.0 celsius")

    def test_output_celsius_to_kelvin(self):
        captured = io.StringIO()
        sys.stdout = captured
        args = argparse.Namespace(
            from_celsius=True,
            to='kelvin',
            temperature=0.0
        )
        cli.main(args)
        sys.stdout = sys.__stdout__()
        self.assertEqual(captured.getvalue().strip(), "0.0 celsius is 273.15 kelvin")

    def test_output_fahrenheit_to_kelvin(self):
        captured = io.StringIO()
        sys.stdout = captured
        args = argparse.Namespace(
            from_fahrenheit=True,
            to='kelvin',
            temperature=32.0
        )
        cli.main(args)
        sys.stdout = sys.__stdout__()
        self.assertEqual(captured.getvalue().strip(), "32.0 fahrenheit is 273.15 kelvin")

    def test_output_kelvin_to_celsius(self):
        captured = io.StringIO()
        sys.stdout = captured
        args = argparse.Namespace(
            from_kelvin=True,
            to='celsius',
            temperature=273.15
        )
        cli.main(args)
        sys.stdout = sys.__stdout__()
        self.assertEqual(captured.getvalue().strip(), "273.15 kelvin is 0.0 celsius")

    def test_output_kelvin_to_fahrenheit(self):
        captured = io.StringIO()
        sys.stdout = captured
        args = argparse.Namespace(
            from_kelvin=True,
            to='fahrenheit',
            temperature=273.15
        )
        cli.main(args)
        sys,stdout = sys.__stdout__()
        self.assertEqual(captured.getvalue().strip(), "273.15 kelvin is 32.0 fahrenheit")