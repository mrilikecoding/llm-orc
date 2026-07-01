"""Shelter module."""

from dog import Dog
from cat import Cat
from animal import Animal


class Shelter:
    """A shelter that holds animals and can produce a roll call."""

    def __init__(self) -> None:
        """Initialise with an empty list of animals."""
        self.animals: list[Animal] = []

    def add(self, animal: Animal) -> None:
        """Add an animal to the shelter.

        Args:
            animal: The animal to add.
        """
        self.animals.append(animal)

    def roll_call(self) -> list[str]:
        """Return the speak output for every animal in the shelter.

        Returns:
            A list of strings, one per animal, in the order they were added.
        """
        return [animal.speak() for animal in self.animals]
