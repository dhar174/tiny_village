import unittest
from tiny_util_funcs import is_numeric, tweener, ClampedIntScore

class TestTinyUtilFuncs(unittest.TestCase):
    def test_is_numeric_valid(self):
        self.assertTrue(is_numeric(10))
        self.assertTrue(is_numeric(10.5))
        self.assertTrue(is_numeric("10"))
        self.assertTrue(is_numeric("10.5"))
        self.assertTrue(is_numeric("-10.5"))

    def test_is_numeric_invalid(self):
        self.assertFalse(is_numeric("abc"))
        self.assertFalse(is_numeric(None))
        self.assertFalse(is_numeric([]))
        self.assertFalse(is_numeric({}))

    def test_tweener_normal(self):
        # input_value=50, max_input=100, start=0, end=10, steps=2
        # Expected: 0 + (10 - 0) * (50 / 100) = 5.0
        self.assertEqual(tweener(50, 100, 0, 10, 2), 5.0)

    def test_tweener_at_max(self):
        # input_value=100, max_input=100, start=0, end=10, steps=2
        # Expected: end = 10
        self.assertEqual(tweener(100, 100, 0, 10, 2), 10)

    def test_tweener_above_max(self):
        # input_value=150, max_input=100, start=0, end=10, steps=2
        # Expected: end = 10
        self.assertEqual(tweener(150, 100, 0, 10, 2), 10)

    def test_tweener_at_zero(self):
        # input_value=0, max_input=100, start=0, end=10, steps=2
        # Expected: 0 + (10 - 0) * (0 / 100) = 0.0
        self.assertEqual(tweener(0, 100, 0, 10, 2), 0.0)

    def test_tweener_reversed(self):
        # input_value=50, max_input=100, start=10, end=0, steps=2
        # Expected: 10 + (0 - 10) * (50 / 100) = 10 - 5 = 5.0
        self.assertEqual(tweener(50, 100, 10, 0, 2), 5.0)

    def test_clamped_int_score_init(self):
        score = ClampedIntScore(min=-5, max=5)
        self.assertEqual(score.min, -5)
        self.assertEqual(score.max, 5)
        self.assertEqual(score.score, 0)

    def test_clamped_int_score_in_range(self):
        score = ClampedIntScore(min=-5, max=5)
        self.assertEqual(score.clamp_score(3), 3)
        self.assertEqual(score.score, 3)

    def test_clamped_int_score_below_min(self):
        score = ClampedIntScore(min=-5, max=5)
        # score < min: -(max(abs(score), 400) / 100)
        # For -10: -(max(10, 400) / 100) = -(400 / 100) = -4.0
        self.assertEqual(score.clamp_score(-10), -4)
        self.assertEqual(score.score, -4.0)

    def test_clamped_int_score_above_max(self):
        score = ClampedIntScore(min=-5, max=5)
        # score > max: max(abs(score), 400) / 100
        # For 10: max(10, 400) / 100 = 400 / 100 = 4.0
        self.assertEqual(score.clamp_score(10), 4)
        self.assertEqual(score.score, 4.0)

    def test_clamped_int_score_large_values(self):
        score = ClampedIntScore(min=-5, max=5)
        # For 500: max(500, 400) / 100 = 500 / 100 = 5.0
        self.assertEqual(score.clamp_score(500), 5)
        self.assertEqual(score.score, 5.0)

        # For -600: -(max(600, 400) / 100) = -6.0
        self.assertEqual(score.clamp_score(-600), -6)
        self.assertEqual(score.score, -6.0)

if __name__ == "__main__":
    unittest.main()
