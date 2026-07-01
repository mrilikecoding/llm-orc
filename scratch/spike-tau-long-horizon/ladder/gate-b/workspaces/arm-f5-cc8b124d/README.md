# Temperature Conversion Library  
A small Python library for converting temperatures between Celsius, Fahrenheit, and Kelvin with a command-line interface.  

## Installation & Usage  
Install via pip (if packaged) or use directly from source.  

### CLI Examples  
```bash  
python cli.py 100 C F  # Converts 100°C to Fahrenheit  
python cli.py 32 F C   # Converts 32°F to Celsius  
```  

### Supported Conversions  
| From    | To      |  
|---------|---------|  
| Celsius | Fahrenheit |  
| Fahrenheit | Celsius |  
| Celsius | Kelvin |  

### CLI Arguments  
`python cli.py <value> <from_unit> <to_unit>`  
Valid units: `C`, `F`, `K`  

## Running Tests  
```bash  
python -m pytest test_converters.py test_cli.py  
# or  
python -m unittest test_converters.py test_cli.py  
```  

## Programmatic Usage  
Import conversion functions from `converters.py`:  
```python  
from converters import celsius_to_fahrenheit  
result = celsius_to_fahrenheit(25)  
print(result)  # Output: 77.0  
```