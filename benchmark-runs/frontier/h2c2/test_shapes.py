import unittest

from shapes import Rectangle


class TestRectangle(unittest.TestCase):
    def setUp(self) -> None:
        self.rect = Rectangle(4.0, 5.0)

    def test_area(self) -> None:
        self.assertEqual(self.rect.area(), 20.0)

    def test_perimeter(self) -> None:
        self.assertEqual(self.rect.perimeter(), 18.0)


if __name__ == "__main__":
    unittest.main()
