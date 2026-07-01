class Rectangle:
    """A class representing a rectangle with width and height."""
    
    def __init__(self, width: float, height: float):
        """Initialize a new Rectangle with given width and height."""
        self.width = width
        self.height = height
    
    def area(self) -> float:
        """Calculate and return the area of the rectangle."""
        return self.width * self.height
    
    def perimeter(self) -> float:
        """Calculate and return the perimeter of the rectangle."""
        return 2 * (self.width + self.height)