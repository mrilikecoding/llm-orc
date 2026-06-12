import io
import sys
import unittest

from cli import main

class TestCLI(unittest.TestCase):
    def test_celsius_to_fahrenheit(self):
        captured_output = io.StringIO()
        sys.stdout = captured_output
        main(['celsius_to_fahrenheit', '25'])
        sys.stdout = sys.__stdout__
        self.assertEqual(captured_output.getvalue().strip(), '77.0')

    def test_fahrenheit_to_celsius(self):
        captured_output = io.StringIO()
        sys.stdout = captured_output
        main(['fahrenheit_to_celsius', '86'])
        sys.stdout = sys.__stdout__
        self.assertEqual(captured_output.getvalue().strip(), '30.0')

    def test_celsius_to_kelvin(self):
        captured_output = io.StringIO()
        sys.stdout = captured_output
        main(['celsius_to_kelvin', '30'])
        sys.stdout = sys.__stdout__
        self.assertEqual(captured_output.getvalue().strip(), '303.15')