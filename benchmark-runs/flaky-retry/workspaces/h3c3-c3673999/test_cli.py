import unittest
import sys
from io import StringIO
from cli import main
from converters import celsius_to_fahrenheit, celsius_to_kelvin

class TestCLI(unittest.TestCase):
    def setUp(self):
        self.captured_output = StringIO()
        sys.stdout = self.captured_output

    def tearDown(self):
        sys.stdout = sys.__stdout__
        self.captured_output.close()

    def test_to_fahrenheit(self):
        sys.argv = ['cli.py', '30', '--to-fahrenheit']
        main()
        expected = str(celsius_to_fahrenheit(30))
        self.assertEqual(self.captured_output.getvalue().strip(), expected)

    def test_to_kelvin(self):
        sys,argv = ['cli.py', '30', '--to-kelvin']
        main()
        expected = str(celsius_to_kelvin(30))
        self.assertEqual(self.captured_output.getvalue().strip(), expected)

    def test_no_flag(self):
        sys.argv = ['cli.py', '30']
        with self.assertRaises(SystemExit):
            main()

    def test_both_flags(self):
        sys.argv = ['cli.py', '30', '--to-fahrenheit', '--to-kelvin']
        with self.assertRaises(SystemExit):
            main()