# Temperature Conversion Library

## Overview
This library provides utilities for temperature unit conversion between Celsius, Fahrenheit, and Kelvin. The implementation is split into two modules: `converters.py` for core conversion logic and `cli.py` for command-line interface support.

## converters.py
The `converters.py` module contains three primary conversion functions:

- **celsius_to_fahrenheit(celsius)**  
  Converts a temperature value from Celsius to Fahrenheit.

- **fahrenheit_to_celsius(fahrenheit)**  
  Converts a temperature value from Fahrenheit to Celsius.

- **celsius_to_kelvin(celsius)**  
  Converts a temperature value from Celsius to Kelvin.

These functions form the foundation of the library's conversion capabilities.

## cli.py
The `cli.py` module provides a command-line interface through the `main()` function. This entry point allows users to interactively convert temperatures using the core functions from `converters.py`. The CLI supports input via standard input and outputs the converted values to standard output.