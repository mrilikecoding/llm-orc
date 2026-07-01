"""Dog module."""

from animal import Animal


class Dog(Animal):
    """A dog that barks."""

    def speak(self) -> str:
        """Return the dog's bark.

        Returns:
            A string representing the dog's vocalisation.
        """
        return f"{self.name} says: Woof!"
