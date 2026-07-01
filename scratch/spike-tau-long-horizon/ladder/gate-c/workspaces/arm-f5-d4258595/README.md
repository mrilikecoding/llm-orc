# Temperature Conversion Library  

A simple Python library for converting temperatures between Celsius, Fahrenheit, and Kelvin.  

## Usage  

### CLI Tool  
Run the command-line interface with:  
```bash
python cli.py --from <UNIT> --to <UNIT> <VALUE>
```  

**Examples:**  
```bash
python cli.py --from celsius --to fahrenheit 100
python cli.py --fahrenheit --to celsius 212
python cli.py --from celsius --to kelvin 0
```  

### Supported Units  
- `celsius` (°C)  
- `fahrenheit` (°F)  
- `kelvin` (K)  

## Testing  
Run tests with `pytest`:  
```bash
pytest test_converters.py test_cli.py
```  

Ensure all test cases pass, including edge cases for invalid units and missing arguments.