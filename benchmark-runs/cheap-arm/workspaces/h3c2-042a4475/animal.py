class Animal:
    """A base class representing a generic animal."""

    def __init__(self, name: str) -> None:
        """Initialize the animal with a name.

        Args:
            name: The name of the animal.
        """
        self.name = name

    def speak(self) -> str:
        """Return the sound this animal makes.

        Returns:
            A string representing the animal's sound.
        """
        return "..."
