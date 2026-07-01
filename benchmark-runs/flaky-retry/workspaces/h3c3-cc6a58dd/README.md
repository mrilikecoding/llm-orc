# Temperature Conversion Package

Convert temperatures between Celsius, Fahrenheit, and Kelvin using a CLI.

## CLI Usage
```bash
python cli.py <value> --to-fahrenheit
python cli.py <value> --to-kelvin
```

Examples:
- Convert 100°C to Fahrenheit: `python cli.py 100 --to-fahrenheit`
- Convert 0°C to Kelvin: `python cli.py 0 --to-kelvin`

## Package Structure
- `converters.py`: Core conversion functions
- `cli.py`: Command-line interface
- `test_converters.py`: Unit tests for converters
- `test_cli.py`: Unit tests for CLI

Requires a numeric value and one of the mutually exclusive flags. Input is assumed to be in Celsius.