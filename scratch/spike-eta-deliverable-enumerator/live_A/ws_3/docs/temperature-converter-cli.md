

# Temperature Conversion Tool Documentation

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Parameter Explanations](#parameter-explanations)
4. [Error Handling](#error-handling)
5. [Integration Examples](#integration-examples)

---

## Installation

### Prerequisites

- Python 3.7 or higher

### Install from Source

Clone the repository and install:

```bash
cd /Users/nathangreen/Development/eddi-lab/llm-orc/scratch/spike-eta-deliverable-enumerator/live_A/ws_3
pip install -e .
```

Alternatively, ensure the project directory is in your Python path:

```bash
export PYTHONPATH="/Users/nathangreen/Development/eddi-lab/llm-orc/scratch/spike-eta-deliverable-enumerator/live_A/ws_3:$PYTHONPATH"
```

---

## Usage

### Command-Line Interface

The CLI allows conversions between Celsius, Fahrenheit, and Kelvin.

#### Basic Syntax

```bash
python cli.py <value> <from_unit> <to_unit>
```

#### Conversion Modes

**Mode 1: Celsius to Fahrenheit**

```bash
python cli.py 100 C F
# Output: 212.0

python cli.py 0 C F
# Output: 32.0

python cli.py -40 C F
# Output: -40.0
```

**Mode 2: Fahrenheit to Celsius**

```bash
python cli.py 32 F C
# Output: 0.0

python cli.py 212 F C
# Output: 100.0
```

**Mode 3: Celsius to Kelvin**

```bash
python cli.py 0 C K
# Output: 273.15

python cli.py 100 C K
# Output: 373.15
```

Additional modes supported via `convert_temperature()` function:

```bash
python cli.py 273.15 K C
# Output: 0.0

python cli.py 273.15 K F
# Output: 32.0

python cli.py 32 F K
# Output: 273.15
```

---

## Parameter Explanations

### Core Functions

#### `temp_convert.celsius_to_fahrenheit(celsius: float) -> float`

Converts a temperature value from Celsius to Fahrenheit.

| Parameter | Type | Description |
|------------|------|-------------|
| `celsius` | float | Temperature in degrees Celsius |

| Return | Type | Description |
|--------|------|-------------|
| result | float | Temperature in degrees Fahrenheit |

#### `temp_convert.fahrenheit_to_celsius(fahrenheit: float) -> float`

Converts a temperature value from Fahrenheit to Celsius.

| Parameter | Type | Description |
|------------|------|-------------|
| `fahrenheit` | float | Temperature in degrees Fahrenheit |

| Return | Type | Description |
|--------|------|-------------|
| result | float | Temperature in degrees Celsius |

#### `temp_convert.celsius_to_kelvin(celsius: float) -> float`

Converts a temperature value from Celsius to Kelvin.

| Parameter | Type | Description |
|------------|------|-------------|
| `celsius` | float | Temperature in degrees Celsius |

| Return | Type | Description |
|--------|------|-------------|
| result | float | Temperature in Kelvin |

#### `cli.kelvin_to_celsius(kelvin: float) -> float`

Converts a temperature value from Kelvin to Celsius.

| Parameter | Type | Description |
|------------|------|-------------|
| `kelvin` | float | Temperature in Kelvin |

| Return | Type | Description |
|--------|------|-------------|
| result | float | Temperature in degrees Celsius |

#### `cli.convert_temperature(value: float, from_unit: str, to_unit: str) -> float`

Universal conversion function supporting all unit combinations.

| Parameter | Type | Description | Valid Values |
|-----------|------|-------------|--------------|
| `value` | float | Temperature value to convert | Any real number |
| `from_unit` | str | Source temperature unit | "C", "F", "K" |
| `to_unit` | str | Target temperature unit | "C", "F", "K" |

| Return | Type | Description |
|--------|------|-------------|
| result | float | Converted temperature value |

---

## Error Handling

### Common Errors and Solutions

#### Invalid Unit Error

If an invalid unit is provided, the CLI displays an error message:

```bash
python cli.py 100 C X
# Error: Invalid unit. Use C (Celsius), F (Fahrenheit), or K (Kelvin).
```

#### Missing Arguments

```bash
python cli.py 100 C
# Error: Incorrect number of arguments.
# Usage: python cli.py <value> <from_unit> <to_unit>
```

#### Type Errors

Ensure numeric values are provided:

```bash
python cli.py abc C F
# Error: Value must be a number.
```

### Program Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error (invalid input, missing arguments, etc.) |

---

## Integration Examples

### Importing Functions

```python
import sys
sys.path.insert(0, '/Users/nathangreen/Development/eddi-lab/llm-orc/scratch/spike-eta-deliverable-enumerator/live_A/ws_3')

from temp_convert import celsius_to_fahrenheit, fahrenheit_to_celsius, celsius_to_kelvin
from cli import kelvin_to_celsius, convert_temperature

# Convert 100 Celsius to Fahrenheit
result = celsius_to_fahrenheit(100.0)
print(f"100°C = {result}°F")  # Output: 100°C = 212.0°F

# Convert 32 Fahrenheit to Celsius
result = fahrenheit_to_celsius(32.0)
print(f"32°F = {result}°C")  # Output: 32°F = 0.0°C

# Convert 0 Celsius to Kelvin
result = celsius_to_kelvin(0.0)
print(f"0°C = {result}K")  # Output: 0°C = 273.15K

# Convert 273.15 Kelvin to Celsius
result = kelvin_to_celsius(273.15)
print(f"273.15K = {result}°C")  # Output: 273.15K = 0.0°C

# Universal conversion function
result = convert_temperature(100, "C", "K")
print(f"100°C = {result}K")  # Output: 100°C = 373.15K
```

### Using the Main CLI Function Programmatically

```python
import sys
sys.path.insert(0, '/Users/nathangreen/Development/eddi-lab/llm-orc/scratch/spike-eta-deliverable-enumerator/live_A/ws_3')

from cli import main

# Simulate command line arguments
sys.argv = ['cli.py', '100', 'C', 'F']

try:
    main()
except SystemExit as e:
    if e.code == 0:
        print("Conversion successful")
    else:
        print(f"Conversion failed with exit code: {e.code}")
```

### Batch Conversion Script

```python
import sys
sys.path.insert(0, '/Users/nathangreen/Development/eddi-lab/llm-orc/scratch/spike-eta-deliverable-enumerator/live_A/ws_3')

from cli import convert_temperature

def batch_convert(conversions):
    results = []
    for value, from_unit, to_unit in conversions:
        try:
            result = convert_temperature(value, from_unit, to_unit)
            results.append(f"{value}{from_unit} = {result}{to_unit}")
        except Exception as e:
            results.append(f"Error converting {value}{from_unit} to {to_unit}: {e}")
    return results

conversions = [
    (0, "C", "F"),
    (100, "C", "F"),
    (32, "F", "C"),
    (212, "F", "C"),
    (0, "C", "K"),
    (100, "C", "K"),
    (273.15, "K", "C"),
    (373.15, "K", "C"),
]

for line in batch_convert(conversions):
    print(line)
```

### Building a Simple Web Service

```python
from flask import Flask, request, jsonify
import sys
sys.path.insert(0, '/Users/nathangreen/Development/eddi-lab/llm-orc/scratch/spike-eta-deliverable-enumerator/live_A/ws_3')

from cli import convert_temperature

app = Flask(__name__)

@app.route('/convert', methods=['GET'])
def convert():
    try:
        value = float(request.args.get('value'))
        from_unit = request.args.get('from')
        to_unit = request.args.get('to')
        
        result = convert_temperature(value, from_unit, to_unit)
        return jsonify({'result': result, 'unit': to_unit})
    except ValueError:
        return jsonify({'error': 'Invalid numeric value'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
```

Example usage:

```bash
curl "http://localhost:5000/convert?value=100&from=C&to=F"
# {"result": 212.0, "unit": "F"}
```

### Unit Testing Integration

```python
import sys
sys.path.insert(0, '/Users/nathangreen/Development/eddi-lab/llm-orc/scratch/spike-eta-deliverable-enumerator/live_A/ws_3')

import unittest
from temp_convert import celsius_to_fahrenheit, fahrenheit_to_celsius, celsius_to_kelvin
from cli import kelvin_to_celsius, convert_temperature

class TestTemperatureConversion(unittest.TestCase):
    
    def test_celsius_to_fahrenheit(self):
        self.assertEqual(celsius_to_fahrenheit(0), 32)
        self.assertEqual(celsius_to_fahrenheit(100), 212)
    
    def test_fahrenheit_to_celsius(self):
        self.assertEqual(fahrenheit_to_celsius(32), 0)
        self.assertEqual(fahrenheit_to_celsius(212), 100)
    
    def test_celsius_to_kelvin(self):
        self.assertEqual(celsius_to_kelvin(0), 273.15)
        self.assertEqual(celsius_to_kelvin(100), 373.15)
    
    def test_kelvin_to_celsius(self):
        self.assertEqual(kelvin_to_celsius(273.15), 0)
        self.assertEqual(kelvin_to_celsius(373.15), 100)
    
    def test_convert_temperature(self):
        self.assertEqual(convert_temperature(100, "C", "F"), 212)
        self.assertEqual(convert_temperature(32, "F", "C"), 0)
        self.assertEqual(convert_temperature(0, "C", "K"), 273.15)

if __name__ == '__main__':
    unittest.main()
```

---

## Additional Notes

- The tool uses the following formula conversions:
  - °F = (°C × 9/5) + 32
  - °C = (°F - 32) × 5/9
  - K = °C + 273.15
  - °C = K - 273.15
- All functions return float values
- Absolute zero (0K / -273.15°C / -459.67°F) is a valid input
- The -40° point is the same in both Celsius and Fahrenheit (the equal point)