import math

"""A Calculator class providing basic arithmetic operations and square root functionality."""

class Calculator:
    def add(self, a: float, b: float) -> float:
        """Adds two numbers and returns the result."""
        return a + b

    def subtract(self, a: float, b: float) -> float:
        """Subtracts two numbers and returns the result."""
        return a - b

    def multiply(self, a: float, b: float) -> float:
        """Multiplies two numbers and returns the result."""
        return a * b

    def divide(self, a: float, b: float) -> float:
        """Divides two numbers and returns the result."""
        if b == 0:
            raise ValueError("Division by zero is not allowed.")
        return a / b

    def power(self, a: float, b: float) -> float:
        """Returns the result of raising a to the power of b."""
        return a ** b

    def sqrt(self, x: float) -> float:
        """Returns the square root of a number."""
        if x < 0:
            raise ValueError("Cannot take square root of negative number.")
        return math.sqrt(x)