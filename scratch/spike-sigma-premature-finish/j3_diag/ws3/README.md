```markdown
# Temperature Converter CLI

This CLI tool provides functions to convert temperatures between Celsius, Fahrenheit, and Kelvin.

## Installation
```bash
pip install .
```

## Usage

### Conversion Commands
- Convert Celsius to Fahrenheit:
  ```bash
  python -m converters celsius_to_fahrenheit 25
  ```
- Convert Fahrenheit to Celsius:
  ```bash
  python -m converters fahrenheit_to_celsius 77
  ```
- Convert Celsius to Kelvin:
  ```bash
  python -m converters celsius_to_kelvin 25
  ```

### Batch Conversion
Convert multiple values at once:
```bash
python -m converters celsius_to_fahrenheit 25 30 35
```

## Tests
The tests in `test_converters.py` ensure correctness for:
- `celsius_to_fahrenheit`
- `fahrenheit_to_celsius`
- `celsius_to_kelvin`

Run tests with:
```bash
python -m unittest test_converters
```

MIT License
```python
# This file is part of the Temperature Converter CLI.
# It is distributed under the MIT License.
```