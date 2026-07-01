"""Tests for Shelter.roll_call."""

import unittest

from shelter import Shelter
from dog import Dog
from cat import Cat


class TestRollCall(unittest.TestCase):
    """Tests for Shelter.roll_call behaviour."""

    def test_empty_shelter_returns_empty_list(self) -> None:
        """roll_call on an empty shelter returns an empty list."""
        shelter = Shelter()
        self.assertEqual(shelter.roll_call(), [])

    def test_single_dog(self) -> None:
        """roll_call with one dog returns that dog's speak output."""
        shelter = Shelter()
        shelter.add(Dog("Rex"))
        self.assertEqual(shelter.roll_call(), ["Rex says: Woof!"])

    def test_single_cat(self) -> None:
        """roll_call with one cat returns that cat's speak output."""
        shelter = Shelter()
        shelter.add(Cat("Whiskers"))
        self.assertEqual(shelter.roll_call(), ["Whiskers says: Meow!"])

    def test_mixed_animals_preserves_order(self) -> None:
        """roll_call returns outputs in insertion order for mixed animals."""
        shelter = Shelter()
        shelter.add(Dog("Buddy"))
        shelter.add(Cat("Luna"))
        shelter.add(Dog("Max"))
        self.assertEqual(
            shelter.roll_call(),
            ["Buddy says: Woof!", "Luna says: Meow!", "Max says: Woof!"],
        )

    def test_multiple_dogs(self) -> None:
        """roll_call works correctly with multiple dogs."""
        shelter = Shelter()
        shelter.add(Dog("Spot"))
        shelter.add(Dog("Fido"))
        self.assertEqual(
            shelter.roll_call(),
            ["Spot says: Woof!", "Fido says: Woof!"],
        )

    def test_multiple_cats(self) -> None:
        """roll_call works correctly with multiple cats."""
        shelter = Shelter()
        shelter.add(Cat("Mittens"))
        shelter.add(Cat("Shadow"))
        self.assertEqual(
            shelter.roll_call(),
            ["Mittens says: Meow!", "Shadow says: Meow!"],
        )


if __name__ == "__main__":
    unittest.main()
