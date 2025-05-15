# File: tests/test_model.py
import unittest
from ascii_globe.model import AsciiMapModel

class TestAsciiMapModel(unittest.TestCase):
    def test_empty(self):
        # sanity check: missing shapefile should raise an exception
        with self.assertRaises(Exception):
            AsciiMapModel("nonexistent.shp")

    # further tests can import actual shapefile fixture and compare output

if __name__ == "__main__":
    unittest.main()