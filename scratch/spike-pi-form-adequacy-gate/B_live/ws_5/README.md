# Temperature Conversion CLI

A command-line tool for converting temperatures between Celsius, Fahrenheit, and Kelvin.

## Usage
```bash
python -m cli [value] [from_unit] [to_unit]
```

## Examples
```bash
python -m cli 32 celsius fahrenheit
python -m cli 98.6 fahrenheit celsius
python -m cli 0 celsius kelvin
```

## Supported Units
- Celsius (`celsius`)
- Fahrenheit (`fahrenheit`)
- Kelvin (`kelvin`)

Note: The tool supports the following conversions:
- Celsius ↔ Fahrenheit
- Celsius ↔ Kelvin