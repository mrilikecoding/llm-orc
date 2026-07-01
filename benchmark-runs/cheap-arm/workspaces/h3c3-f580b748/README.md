# Temperature Conversion Package

A simple Python package for converting temperatures between Celsius, Fahrenheit, and Kelvin.

## Installation
Install the package using pip:

```bash
pip install .
```

Or, if you're developing locally:

```bash
pip install -e .
```

## CLI Usage
Convert temperature values using the command-line interface:

```bash
python cli.py 25 --to-fahrenheit
python cli.py 32 --to-kelvin
```

## Testing
Run the tests using Python's unittest module:

```bash
python -m unittest test_converters.py
python -m unittest test_cli.py
```