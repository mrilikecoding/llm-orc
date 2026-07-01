import unittest
from unittest import mock
import sys
import io
from cli import main

class TestCli(unittest.TestCase):
    def test_convert_to_fahrenheit_normal(self):
        with mock.patch('sys.argv', ['cli.py', '25', '--to-fahrenheit']):
            with io.StringIO() as buf, mock.patch('sys.stdout', buf):
                main()
                output = buf.getvalue().strip()
                self.assertEqual(output, "77.0°F")

    def test_convert_to_fahrenheit_edge_0(self):
        with mock.patch('sys.argv', ['cli.py', '0', '--to-fahrenheit']):
            with io.StringIO() as buf, mock.patch('sys.stdout', buf):
                main()
                output = buf.getvalue().strip()
                self.assertEqual(output, "32.0°F")

    def test_convert_to_fahrenheit_edge_neg40(self):
        with mock.patch('sys.argv', ['cli.py', '-40', '--to-fahrenheit']):
            with io.StringIO() as buf, mock.patch('sys.stdout', buf):
                main()
                output = buf.getvalue().strip()
                self.assertEqual(output, "-40.0°F")

    def test_convert_to_fahrenheit_edge_100(self):
        with mock.patch('sys.argv', ['cli.py', '100', '--to-fahrenheit']):
            with io.StringIO() as buf, mock.patch('sys.stdout', buf):
                main()
                output = buf.getvalue().strip()
                self.assertEqual(output, "212.0°F")

    def test_convert_to_kelvin_normal(self):
        with mock.patch('sys.argv', ['cli.py', '25', '--to-kelvin']):
            with io.StringIO() as buf, mock.patch('sys.stdout', buf):
                main()
                output = buf.getvalue().strip()
                self.assertEqual(output, "298.15K")

    def test_convert_to_kelvin_edge_0(self):
        with mock.patch('sys.argv', ['cli.py', '0', '--to-kelvin']):
            with io.StringIO() as buf, mock.patch('sys.stdout', buf):
                main()
                output = buf.getvalue().strip()
                self.assertEqual(output, "273.15K")

    def test_convert_to_kelvin_edge_neg40(self):
        with mock.patch('sys.argv', ['cli.py', '-40', '--to-kelvin']):
            with io.StringIO() as buf, mock.patch('sys.stdout', buf):
                main()
                output = buf.getvalue().strip()
                self.assertEqual(output, "133.15K")

    def test_convert_to_kelvin_edge_100(self):
        with mock.patch('sys.argv', ['cli.py', '100', '--to-kelvin']):
            with io.StringIO() as buf, mock.patch('sys.stdout', buf):
                main()
                output = buf.getvalue().strip()
                self.assertEqual(output, "373.15K")

    def test_missing_value_error(self):
        with mock.patch('sys.argv', ['cli.py']):
            with self.assertRaises(SystemExit):
                main()

    def test_extra_flags_error(self):
        with mock.patch('sys.argv', ['cli.py', '25', '--to-fahrenheit', '--to-kelvin']):
            with self.assertRaises(SystemExit):
                main()

    def test_non_numeric_value_error(self):
        with mock.patch('sys.argv', ['cli.py', 'abc', '--to-fahrenheit']):
            with self.assertRaises(ValueError):
                main()