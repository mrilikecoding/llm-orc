

# Temperature Conversion Tool

A command-line utility for converting temperatures between Celsius, Fahrenheit, and Kelvin.

## Installation

1. Ensure Python 3.7 or higher is installed on your system.

2. Clone or download this repository to your local machine.

3. Navigate to the project directory:

   ```bash
   cd /path/to/project
   ```

4. No additional dependencies are required. The tool uses only the Python standard library.

## Usage

The CLI tool is implemented in `temperature_cli.py` and providestemperature conversion functionality through the `main()` function.

### Basic Syntax

Run the CLI tool using Python:

```bash
python temperature_cli.py <value> <from_unit> <to_unit>
```

### Arguments

- `value` - The numerical temperature value to convert
- `from_unit` - The source temperature unit (celsius, fahrenheit, or kelvin)
- `to_unit` - The target temperature unit (celsius, fahrenheit, or kelvin)

## Available Conversions

The following conversion functions are available in the underlying module (`temperature.py`):

- `celsius_to_fahrenheit(celsius)` - Convert Celsius to Fahrenheit
- `fahrenheit_to_celsius(fahrenheit)` - Convert Fahrenheit to Celsius
- `celsius_to_kelvin(celsius)` - Convert Celsius to Kelvin

The CLI (`convert_temperature`) uses these functions to handle conversions between the supported units.

## Examples

### Convert Celsius to Fahrenheit

```bash
python temperature_cli.py 100 celsius fahrenheit
```

Output: `212.0`

### Convert Fahrenheit to Celsius

```bash
python temperature_cli.py 32 fahrenheit celsius
```

Output: `0.0`

### Convert Celsius to Kelvin

```bash
python temperature_cli.py 0 celsius kelvin
```

Output: `273.15`

### Convert a Negative Temperature

```bash
python temperature_cli.py -40 celsius fahrenheit
```

Output: `-40.0`

## Running Tests

To verify the tool works correctly, run the included test suite:

```bash
python -m pytest tests/ -v
```

Or run individual test files:

```bash
python -m pytest tests/test_temperature.py -v
python -m pytest tests/test_cli.py -v
```

## Unit Abbreviations

When specifying units, use the following abbreviations:

- `celsius` or `C`
- `fahrenheit` or `F`
- `kelvin` or `K`

The tool handles unit matching based on the `convert_temperature` function implementation.