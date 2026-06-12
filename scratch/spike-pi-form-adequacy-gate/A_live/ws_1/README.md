# Temperature Unit Converter CLI Tool

## Description
A command-line interface tool for converting temperatures between Celsius, Fahrenheit, and Kelvin units. Supports direct conversions between these scales.

## Usage
Run the tool using Python with the following syntax:

```bash
python cli.py --value <VALUE> --from <FROM_UNIT> --to <TO_UNIT>
```

### Required Arguments
- `--value`: The temperature value to convert (required)
- `--from`: Source unit (required, must be one of: celsius, fahrenheit, kelvin)
- `--to`: Target unit (required, must be one of: celsius, fahrenheit, kelvin)

### Supported Conversions
The tool supports these direct conversions:
- Celsius ↔ Fahrenheit
- Celsius ↔ Kelvin
- Fahrenheit ↔ Celsius

### Example
Convert 25°C to Fahrenheit:
```bash
python cli.py --value 25 --from celsius --to fahrenheit
```

Output:
```
25 celsius is equal to 77.0 fahrenheit
```

## Notes
- Only the specified conversions are supported. Other unit pairs will show an error.
- All values are treated as floating-point numbers.