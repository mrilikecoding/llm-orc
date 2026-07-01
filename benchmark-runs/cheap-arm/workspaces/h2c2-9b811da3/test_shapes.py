import unittest
from shapes import Rectangle

class TestRectangle(unittest.TestCase):
    def test_area_and_perimeter(self):
        rect = Rectangle(5.0, 3.0)
        self.assertEqual(rect.area(), 15.0)
        self.assertEqual(rect.perimeter(), 16.0)
    
    def test_another_area_and_perimeter(self):
        rect = Rectangle(4.0, 2.0)
        self.assertEqual(rect.area(), 8.0)
        self.assertEqual(rect.perimeter(), 12.0)

if __name__ == '__main__':
    unittest.main()