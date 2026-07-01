# Temperature Converters

A small Python package for converting temperatures between Celsius, Fahrenheit, and Kelvin.

## Files

- `converters.py` — pure conversion functions
- `cli.py` — command-line interface
- `test_converters.py` — unit tests for converters
- `test_cli.py` — unit tests for the CLI

## CLI Usage

Run `cli.py` directly with Python. Pass a numeric value and one conversion flag.

### Convert Celsius to Fahrenheit

```
python cli.py 100 --to-fahrenheit
212.0
```

### Convert Celsius to Kelvin

```
python cli.py 0 --to-kelvin
273.15
```

### Convert Fahrenheit to Celsius

```
python cli.py 212 --to-celsius
100.0
```

### Flags

- `--to-fahrenheit` — interpret value as Celsius, output Fahrenheit
- `--to-kelvin` — interpret value as Celsius, output Kelvin
- `--to-celsius` — interpret value as Fahrenheit, output Celsius

Exactly one flag is required. Passing more than one or none exits with an error.

## Running Tests

```
python -m unittest discover .
```
