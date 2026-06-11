# Temperature Conversion Library

## Installation

Install the library using pip:

```bash
pip install temperature-converter
```

## Usage

The library provides a CLI tool for temperature conversions. Supported conversion types:

- `celsius_to_fahrenheit`
- `fahrenheit_to_celsius`
- `celsius_to_kelvin`

### Example Commands

```bash
python cli_tool.py celsius_to_fahrenheit 25
# Result: 77.0

python cli_tool.py fahrenheit_to_celsius 77
# Result: 25.0

python cli_tool.py celsius_to_kelvin 0
# Result: 273.15
```

## Available Functions

The library includes the following conversion functions:

1. **`celsius_to_fahrenheit(celsius)`**
   - Converts Celsius to Fahrenheit.
   - Formula: $ F = (C \times \frac{9}{5}) + 32 $
   - Parameters: `celsius` (float)
   - Returns: float (Fahrenheit value)

2. **`fahrenheit_to_celsius(fahrenheit)`**
   - Converts Fahrenheit to Celsius.
   - Formula: $ C = (F - 32) \times \frac{5}{9} $
   - Parameters: `fahrenheit` (float)
   - Returns: float (Celsius value)

3. **`celsius_to_kelvin(celsius)`**
   - Converts Celsius to Kelvin.
   - Formula: $ K = C + 273.15 $
   - Parameters: `celsius` (float)
   - Returns: float (Kelvin value)

## Testing

Unit tests are located in `test_converters.py`. To run the tests, ensure `pytest` is installed and execute:

```bash
pytest test_converters.py
```

## License

This library is released under the MIT License. See [LICENSE](LICENSE) for details.