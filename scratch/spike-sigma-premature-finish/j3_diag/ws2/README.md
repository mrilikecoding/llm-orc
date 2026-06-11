# Temperature Conversion Library

A simple library for converting temperatures between Celsius, Fahrenheit, and Kelvin.

## Installation

To install the library, run:

```bash
pip install .
```

## Usage

The library includes a CLI tool for temperature conversion. Example usage:

```bash
python cli.py 100 --from celsius --to fahrenheit
# Output: 100 celsius is equal to 212.0 fahrenheit

python cli.py 32 --from fahrenheit --to celsius
# Output: 32 fahrenheit is equal to 0.0 celsius

python cli.py 0 --from celsius --to kelvin
# Output: 0 celsius is equal to 273.15 kelvin
```

Supported conversions:
- Celsius → Fahrenheit
- Celsius → Kelvin
- Fahrenheit → Celsius

## Project Structure

- `converters.py`: Contains the core temperature conversion functions (`celsius_to_fahrenheit`, `fahrenheit_to_celsius`, `celsius_to_kelvin`).
- `cli.py`: Implements the command-line interface using `argparse`.
- `test_converters.py`: Unit tests for the conversion functions.
- `test_cli.py`: Tests for the CLI tool, including error handling for invalid units and unsupported conversions.