# Temperature Converter Library
A Python library for converting temperatures between Celsius, Fahrenheit, Kelvin, and Rankine scales with a command-line interface.

## Installation
```bash
pip install temperature-converter
```

## Usage Examples
```python
from converters import celsius_to_fahrenheit, fahrenheit_to_kelvin

# Convert 25°C to Fahrenheit
print(celsius_to_fahrenheit(25))  # Output: 77.0

# Convert 98.6°F to Kelvin
print(fahrenheit_to_kelvin(98.6))  # Output: 310.15
```

## CLI Usage
```bash
temperature-converter --from CELSIUS --to FAHRENHEIT --value 25
temperature-converter --from FAHRENHEIT --to KELVIN --value 98.6
temperature-converter --from KELVIN --to RANKINE --value 300
temperature-converter --help
```

## Testing
```bash
pip install pytest
pytest
```

## Contribution Guidelines
1. Fork the repository
2. Create a new branch for each feature
3. Submit pull requests with clear descriptions
4. Follow PEP8 style guidelines
5. Add tests for all new functionality
6. Update documentation for new features

## License
MIT License

Copyright (c) [YEAR] [NAME]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.