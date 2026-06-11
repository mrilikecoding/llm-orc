

import sys
sys.path.insert(0, '/Users/nathangreen/Development/eddi-lab/llm-orc/scratch/spike-eta-deliverable-enumerator/live_A/ws_3')
from temp_convert import celsius_to_fahrenheit, fahrenheit_to_celsius, celsius_to_kelvin


# Test celsius_to_fahrenheit
assert celsius_to_fahrenheit(0) == 32, "0°C should be 32°F"
assert celsius_to_fahrenheit(100) == 212, "100°C should be 212°F"
assert celsius_to_fahrenheit(-40) == -40, "-40°C should be -40°F (edge case: equal point)"
assert celsius_to_fahrenheit(-273.15) == -459.67, "Absolute zero in Celsius"


# Test fahrenheit_to_celsius
assert fahrenheit_to_celsius(32) == 0, "32°F should be 0°C"
assert fahrenheit_to_celsius(212) == 100, "212°F should be 100°C"
assert fahrenheit_to_celsius(-40) == -40, "-40°F should be -40°C (edge case: equal point)"
assert fahrenheit_to_celsius(-459.67) == -273.15, "Absolute zero in Fahrenheit"


# Test celsius_to_kelvin
assert celsius_to_kelvin(0) == 273.15, "0°C should be 273.15K"
assert celsius_to_kelvin(100) == 373.15, "100°C should be 373.15K"
assert celsius_to_kelvin(-273.15) == 0, "-273.15°C should be 0K (absolute zero)"
assert celsius_to_kelvin(-40) == 233.15, "-40°C should be 233.15K"


print("All tests passed!")