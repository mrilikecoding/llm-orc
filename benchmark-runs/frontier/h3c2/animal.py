"""Base animal module."""


class Animal:
    """An abstract base animal with a name and a speak method."""

    def __init__(self, name: str) -> None:
        """Initialise with a name.

        Args:
            name: The animal's name.
        """
        self.name = name

    def speak(self) -> str:
        """Return the sound the animal makes.

        Returns:
            A string representing the animal's vocalisation.
        """
        raise NotImplementedError
