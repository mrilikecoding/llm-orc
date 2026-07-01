# Temperature Converter CLI

## Usage
Convert Celsius values to Fahrenheit or Kelvin using the command line interface.

### Basic Command
```bash
python cli.py <value> [--to-fahrenheit | --to-kelvin]
```

### Example Commands
```bash
# Convert 25°C to Fahrenheit
python cli.py 25 --to-fahrenheit
# Output: 77.0°F

# Convert 100°C to Kelvin
python cli.py 100 --to-kelvin
# Output: 373.15K
```

### Notes
- The positional argument `<value>` must be a numeric Celsius value
- Use exactly one of the mutually exclusive flags:
  - `--to-fahrenheit` for Fahrenheit conversion
  - `--to-kelvin` for Kelvin conversion
- The converter uses precise formulas from `converters.py`