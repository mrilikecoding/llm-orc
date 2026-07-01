class Rectangle:
    """A rectangle defined by width and height."""

    def __init__(self, width: float, height: float) -> None:
        """Initialise the rectangle.

        Args:
            width: The width of the rectangle.
            height: The height of the rectangle.
        """
        self.width = width
        self.height = height

    def area(self) -> float:
        """Return the area of the rectangle."""
        return self.width * self.height

    def perimeter(self) -> float:
        """Return the perimeter of the rectangle."""
        return 2 * (self.width + self.height)
