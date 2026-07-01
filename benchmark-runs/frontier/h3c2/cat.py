"""Cat module."""

from animal import Animal


class Cat(Animal):
    """A cat that meows."""

    def speak(self) -> str:
        """Return the cat's meow.

        Returns:
            A string representing the cat's vocalisation.
        """
        return f"{self.name} says: Meow!"
