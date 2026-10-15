import unittest
from tiny_util_funcs import ClampedIntScore, is_numeric, tweener

class TestTinyUtilFuncs(unittest.TestCase):
    # --- Tests for is_numeric ---
    def test_is_numeric_positive(self):
        self.assertTrue(is_numeric(123))
        self.assertTrue(is_numeric(123.45))
        self.assertTrue(is_numeric("123"))
        self.assertTrue(is_numeric("123.45"))
        self.assertTrue(is_numeric("1e3"))
        self.assertTrue(is_numeric("-5"))

    def test_is_numeric_negative(self):
        self.assertFalse(is_numeric("abc"))
        self.assertFalse(is_numeric("123a"))
        self.assertFalse(is_numeric(""))
        self.assertFalse(is_numeric([]))
        self.assertFalse(is_numeric({}))

    def test_is_numeric_none(self):
        self.assertFalse(is_numeric(None))

    # --- Tests for ClampedIntScore ---
    def test_clamped_int_score_init(self):
        score = ClampedIntScore()
        self.assertEqual(score.min, -4)
        self.assertEqual(score.max, 4)
        self.assertEqual(score.score, 0)

        score_custom = ClampedIntScore(min=-10, max=10)
        self.assertEqual(score_custom.min, -10)
        self.assertEqual(score_custom.max, 10)

    def test_clamped_int_score_within_bounds(self):
        score = ClampedIntScore(min=-5, max=5)
        self.assertEqual(score.clamp_score(3), 3)
        self.assertEqual(score.score, 3)
        self.assertEqual(score.clamp_score(-2), -2)
        self.assertEqual(score.score, -2)
        self.assertEqual(score.clamp_score(0), 0)

    def test_clamped_int_score_outside_bounds(self):
        score = ClampedIntScore(min=-4, max=4)

        # Slightly outside: abs(score) < 400
        # 5 > 4, max(5, 400)/100 = 4.0
        self.assertEqual(score.clamp_score(5), 4)

        # -10 < -4, -(max(10, 400)/100) = -4.0
        self.assertEqual(score.clamp_score(-10), -4)

        # Far outside: abs(score) >= 400
        # 500 > 4, max(500, 400)/100 = 5.0
        self.assertEqual(score.clamp_score(500), 5)

        # -600 < -4, -(max(600, 400)/100) = -6.0
        self.assertEqual(score.clamp_score(-600), -6)

    def test_clamped_int_score_setters(self):
        score = ClampedIntScore()
        score.set_min(-20)
        score.set_max(20)
        self.assertEqual(score.min, -20)
        self.assertEqual(score.max, 20)
        self.assertEqual(score.clamp_score(15), 15)

    def test_clamped_int_score_repr_str(self):
        score = ClampedIntScore(min=-4, max=4)
        score.clamp_score(2)
        self.assertEqual(repr(score), "ClampedIntScore(2, -4, 4)")
        self.assertEqual(str(score), "ClampedIntScore with score 2, min -4, max 4.")

    # --- Tests for tweener ---
    def test_tweener_interpolation(self):
        # start=0, end=100, max_input=10, input=5 -> 50.0
        self.assertAlmostEqual(tweener(5, 10, 0, 100, 10), 50.0)
        # start=10, end=20, max_input=100, input=25 -> 12.5
        self.assertAlmostEqual(tweener(25, 100, 10, 20, 10), 12.5)

    def test_tweener_boundaries(self):
        # input >= max_input
        self.assertEqual(tweener(10, 10, 0, 100, 10), 100)
        self.assertEqual(tweener(15, 10, 0, 100, 10), 100)
        # input = 0
        self.assertEqual(tweener(0, 10, 0, 100, 10), 0)

if __name__ == "__main__":
    unittest.main()
