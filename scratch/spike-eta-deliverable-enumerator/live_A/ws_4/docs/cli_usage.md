

Temperature Conversion Tool (tempconvert)
==========================================

1. Introduction
---------------
tempconvert is a command-line interface (CLI) tool for converting temperatures between Celsius, Fahrenheit, and Kelvin. It provides a simple and efficient way to perform temperature conversions without needing to write Python code or use a graphical interface.

2. Usage Syntax
---------------
The basic syntax for the tempconvert CLI tool is:

    python -m cli [OPTIONS]

Mandatory Options:
  --c2f <value>     Convert Celsius to Fahrenheit
  --f2c <value>    Convert Fahrenheit to Celsius
  --c2k <value>    Convert Kelvin to Celsius

You must specify exactly one conversion flag with a temperature value.

3. Supported Units
------------------
The tool supports three temperature units:

  Celsius (C):     The metric system unit of temperature. Water freezes at 0°C and boils at 100°C at standard atmospheric pressure.
  Fahrenheit (F):  A temperature unit used primarily in the United States. Water freezes at 32°F and boils at 212°F at standard atmospheric pressure.
  Kelvin (K):      The SI base unit of thermodynamic temperature. Absolute zero is 0 K. Water freezes at 273.15 K and boils at 373.15 K.

4. Example Commands
-------------------
Convert 25 degrees Celsius to Fahrenheit:

    python -m cli --c2f 25

Expected output: 77.0°F

Convert 100 degrees Celsius to Kelvin:

    python -m cli --c2k 100

Expected output: 373.15K

5. Error Handling
-----------------
The tool includes robust error handling for invalid inputs:

  - Non-numeric input values will result in an error message indicating that a valid numeric temperature value is required.
  - Invalid or unrecognized flags will produce an error message listing the available options.
  - Missing required arguments will trigger an error prompting the user to provide the necessary input.

When an error occurs, the tool displays a descriptive message to help the user correct their command. Review the error message and ensure that:
  - A valid conversion flag (--c2f, --f2c, --c2k) is provided
  - The temperature value is a valid number (integer or decimal)
  - The command follows the correct syntax as shown in the usage section above