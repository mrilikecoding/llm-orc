```markdown
# Temperature Conversion CLI Tool

A command-line tool for converting temperatures between Celsius, Fahrenheit, and Kelvin.

## Installation

1. Clone the repository
2. Ensure Python is installed

## Usage

Convert temperature values using the CLI:

```bash
python cli.py <value> <from_unit> <to_unit>
```

Examples:
```bash
python cli.py 0 celsius fahrenheit   # 0°C = 32°F
python cli.py 212 fahrenheit celsius # 212°F = 100°C
python cli.py -273.15 celsius kelvin # -273.15°C = 0K
```

Supported conversions:
- Celsius to Fahrenheit
- Fahrenheit to Celsius
- Celsius to Kelvin

## Testing

Run tests with pytest:

```bash
pytest
```

This will execute all unit tests in `test_converters.py` and `test_cli.py`.
```