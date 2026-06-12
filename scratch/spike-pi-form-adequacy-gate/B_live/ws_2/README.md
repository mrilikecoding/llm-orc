```markdown
# Temperature Conversion CLI Tool

## Installation
```bash
pip install .
```

## Usage
Convert temperatures using the following commands:

### Celsius to Fahrenheit
```bash
convert --from celsius --to fahrenheit <value>
```
Example:
```bash
convert --from celsius --to fahrenheit 0
```

### Fahrenheit to Celsius
```bash
convert --from fahrenheit --to celsius <value>
```
Example:
```bash
convert --from fahrenheit --to celsius 212
```

### Celsius to Kelvin
```bash
convert --from celsius --to kelvin <value>
```
Example:
```bash
convert --from celsius --to kelvin 273.15
```

## Conversion Options
- `--from`: Specify the input unit (celsius, fahrenheit)
- `--to`: Specify the output unit (fahrenheit, celsius, kelvin)
- `<value>`: The numerical value to convert

The CLI tool uses the following functions:
- `celsius_to_fahrenheit()`: Converts Celsius to Fahrenheit
- `fahrenheit_to_celsius()`: Converts Fahrenheit to Celsius
- `celsius_to_kelvin()`: Converts Celsius to Kelvin
```