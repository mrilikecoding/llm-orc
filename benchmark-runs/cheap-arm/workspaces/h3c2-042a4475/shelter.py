from dog import Dog
from cat import Cat

class Shelter:
    def __init__(self) -> None:
        """Initialize a Shelter with an empty list of animals."""
        self.animals = []

    def add_animal(self, animal: Dog | Cat) -> None:
        """Add an animal to the shelter."""
        self.animals.append(animal)

    def roll_call(self) -> list[str]:
        """Return a list of strings where each string is the animal's name followed by their speak() output."""
        return [f"{animal.name} {animal.speak()}" for animal in self.animals]