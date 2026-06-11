# temperature-conversion-library

A simple library for converting temperature units between Celsius, Fahrenheit, and Kelvin.

## Installation

Install the package using pip:

```bash
pip install .
```

## Usage

### As a Python library

Import and use the conversion functions:

```python
from converters import celsius_to_fahrenheit, fahrenheit_to_celsius, celsius_to_kelvin

# Example
fahrenheit = celsius_to_fahrenheit(25)
celsius = fahrenheit_to_celsius(77)
kelvin = celsius_to_kelvin(0)
```

### As a CLI tool

Use the command-line interface to convert temperatures:

```bash
python cli.py --temperature <value> --from <unit> --to <unit>
```

**Supported units**: `celsius`, `fahrenheit`, `kelvin`  
**Supported conversions**:
- Celsius → Fahrenheit
- Fahrenheit → Celsius
- Celsius → Kelvin

**Example**:
```bash
python cli.py --temperature 25 --from celsius --to fahrenheit
# Output: 25 celsius is equal to 77.0 fahrenheit.
```

## Testing

Run tests using pytest:

```bash
pytest test_converters.py test_cli.py
```

## Limitations

- This version only supports conversions between Celsius, Fahrenheit, and Kelvin for the following specific pairs:
  - Celsius → Fahrenheit
  - Fahrenheit → Celsius
  - Celsius → Kelvin
- Other conversions (e.g., Fahrenheit → Kelvin) are not implemented.
- The CLI tool explicitly lists unsupported conversions with a message like:
  `"Conversion from {args.from} to {args.to} is not supported"`