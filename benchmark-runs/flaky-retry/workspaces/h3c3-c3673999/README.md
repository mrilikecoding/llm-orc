```markdown
# Temperature Converter CLI

A simple Python package for converting temperatures between Celsius, Fahrenheit, and Kelvin using a command-line interface.

## Installation

Install the package using pip:

```bash
pip install temperature-converter
```

Or, if installing from source:

```bash
pip install .
```

## Usage

The package provides a command-line interface (CLI) tool for converting temperatures. To use it, run:

```bash
python cli.py <value> --to-fahrenheit|--to-kelvin
```

Where `<value>` is the numeric temperature to convert.

## Examples

Convert 100°C to Fahrenheit:
```bash
python cli.py 100 --to-fahrenheit
```

Convert 32°C to Kelvin:
```bash
python cli.py 32 --to-kelvin
```

Flags `--to-fahrenheit` and `--to-kelvin` are mutually exclusive. Only one flag may be used at a time.
```