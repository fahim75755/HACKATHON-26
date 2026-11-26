"""Temporary test added by OkIttehad to check push access. Safe to delete."""
import unittest


class TestPushCheck(unittest.TestCase):
    def test_basic_math(self):
        self.assertEqual(2 + 2, 4)

    def test_string(self):
        self.assertTrue("hackathon".upper().startswith("HACK"))


if __name__ == "__main__":
    unittest.main()
