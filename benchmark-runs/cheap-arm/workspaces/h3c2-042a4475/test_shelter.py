import unittest
from dog import Dog
from cat import Cat
from shelter import Shelter

class TestShelter(unittest.TestCase):
    def setUp(self) -> None:
        """Set up a Shelter with a Dog and Cat for testing."""
        self.shelter = Shelter()
        self.shelter.add_animal(Dog("Buddy"))
        self.shelter.add_animal(Cat("Whiskers"))
    
    def test_roll_call(self) -> None:
        """Test that Shelter.roll_call() returns the expected list of animal sounds."""
        expected = ["Buddy woof", "Whiskers meow"]
        self.assertEqual(self.shelter.roll_call(), expected)

if __name__ == "__main__":
    unittest.main()