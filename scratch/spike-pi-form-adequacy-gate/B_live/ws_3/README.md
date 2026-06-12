```plaintext
# Temperature Conversion Library

A simple Python library for converting temperatures between Celsius, Fahrenheit, and Kelvin.

## Installation

To install the library, run:

```bash
pip install .
```

Ensure you have the package in your Python path or installed in a virtual environment.

## CLI Usage

The library includes a command-line interface for quick conversions. Use the following commands:

### Convert Celsius to Fahrenheit
```bash
convert temp celsius_to_fahrenheit <value>
```

### Convert Fahrenheit to Celsius
```bash
convert temp fahrenheit_to_celsius <value>
```

### Convert Celsius to Kelvin
```bash
convert temp celsius_to_kelvin <value>
```

Replace `<value>` with the temperature value you wish to convert.

## Development

### Running Tests
```bash
python -m pytest test_converters.py
```

Ensure all test cases pass to verify correctness.
```