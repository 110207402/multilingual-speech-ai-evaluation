import unittest

from speech_eval import character_error_rate, diarization_error_rate, pii_scores, role_accuracy


class MetricsTest(unittest.TestCase):
    def test_character_error_rate_ignores_whitespace(self):
        self.assertEqual(character_error_rate("你 好", "你好"), 0.0)

    def test_character_error_rate_counts_substitution(self):
        self.assertEqual(character_error_rate("你好", "你們"), 0.5)

    def test_pii_scores(self):
        result = pii_scores(["A", "B"], ["B", "C"])
        self.assertEqual(result, {"precision": 0.5, "recall": 0.5, "f1": 0.5})

    def test_diarization_error_rate(self):
        self.assertAlmostEqual(diarization_error_rate(1, 2, 3, 100), 0.06)

    def test_role_accuracy(self):
        self.assertEqual(role_accuracy(["agent", "customer"], ["agent", "agent"]), 0.5)


if __name__ == "__main__":
    unittest.main()
