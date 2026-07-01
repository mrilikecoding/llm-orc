# Temperature Conversion Library

## Overview
A simple Python library for converting temperatures between Celsius, Fahrenheit, and Kelvin units. Includes a command-line interface (CLI) for quick conversions.

## Installation
No package installation required. Just need Python 3.

## Usage
Run the CLI tool with:
```bash
python cli.py <value> <from_unit> <to_unit>
```
Supported units: `C` (Celsius), `F` (Fahrenheit), `K` (Kelvin)

## Examples
```bash
python cli.py 100 C F    # Converts 100°C to Fahrenheit
python cli.py 32 F C    # Converts 32°F to Celsius
python cli.py 0 C K     # Converts 0°C to Kelvin
```

## Running Tests
```bash
python -m pytest
```

## API Reference
### `celsius_to_fahrenheit(celsius: float) -> float`
Converts Celsius to Fahrenheit.
```python
celsius_to_fahrenheit(100)  # Returns 212.0
```

### `fahrenheit_to_celsius(fahrenheit: float) -> float`
Converts Fahrenheit to Celsius.
```python
fahrenheit_to_celsius(32)  # Returns 0.0
```

### `celsius_to_kelvin(celsius: float) -> float`
Converts Celsius to Kelvin.
```python
celsius_to_kelvin(0)  # Returns 273.15
```