from animal import Animal

class Cat(Animal):
    """A cat animal that meows."""
    
    def __init__(self, name: str) -> None:
        super().__init__(name)
    
    def speak(self) -> str:
        """Return the cat's meow sound."""
        return "Meow!"