from animal import Animal

class Dog(Animal):
    def __init__(self, name: str) -> None:
        """Initialize a Dog with a name.
        
        Args:
            name (str): The name of the dog.
        """
        super().__init__(name)
    
    def speak(self) -> str:
        """Return the dog's woof sound.
        
        Returns:
            str: The sound the dog makes.
        """
        return f"{self.name} says Woof!"