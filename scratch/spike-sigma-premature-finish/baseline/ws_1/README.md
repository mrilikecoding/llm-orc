# Temperature Conversion Library

## Installation

To install the library, run:

```bash
pip install .
```

Ensure the package is properly set up with a `setup.py` file in your project directory.

## Usage

The library provides a command-line interface (CLI) for temperature conversions. Use the following command format:

```bash
python cli.py <value> <conversion>
```

Where:
- `<value>` is the temperature value to convert (as a float).
- `<conversion>` is the type of conversion (one of: `celsius_to_fahrenheit`, `fahrenheit_to_celsius`, `celsius_to_kelvin`).

## Examples

### Convert Celsius to Fahrenheit
```bash
python cli.py 25 celsius_to_fahrenheit
```
**Output:**
```
Result: 77.0
```

### Convert Fahrenheit to Celsius
```bash
python cli.py 32 fahrenheit_to_celsius
```
**Output:**
```
Result: 0.0
```

### Convert Celsius to Kelvin
```bash
python cli.py 25 celsius_to_kelvin
```
**Output:**
```
Result: 298.15
```

## Available Conversions

- `celsius_to_fahrenheit`: Converts Celsius to Fahrenheit.
- `fahrenheit_to_celsius`: Converts Fahrenheit to Celsius.
- `celsius_to_kelvin`: Converts Celsius to Kelvin.

## Error Handling

The CLI will display an error message if:
- An invalid conversion type is specified.
- Required arguments (`value` or `conversion`) are missing.
- A non-numeric value is provided for the temperature.