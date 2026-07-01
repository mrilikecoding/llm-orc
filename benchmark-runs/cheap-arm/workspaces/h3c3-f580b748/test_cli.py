import sys
from io import StringIO
import unittest
import cli

class TestCLI(unittest.TestCase):
    def test_to_fahrenheit(self):
        sys.argv = ['script', '0', '--to-fahrenheit']
        with StringIO() as fake_stdout:
            sys.stdout = fake_stdout
            cli.main()
            sys.stdout = sys.__stdout__
            self.assertEqual(fake_stdout.getvalue().strip(), '32.0')

    def test_to_kelvin(self):
        sys.argv = ['script', '0', '--to-kelvin']
        with StringIO() as fake_stdout:
            sys.stdout = fake_stdout
            cli.main()
            sys.stdout = sys.__stdout__
            self.assertEqual(fake_stdout.getvalue().strip(), '273.15')

    def test_mutually_exclusive_flags(self):
        sys.argv = ['script', '--to-fahrenheit', '--to-kelvin']
        with self.assertRaises(SystemExit):
            cli.main()

    def test_missing_value_argument(self):
        sys.argv = ['script', '--to-fahrenheit']
        with self.assertRaises(SystemExit):
            cli.main()