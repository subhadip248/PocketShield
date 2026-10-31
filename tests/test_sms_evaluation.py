"""Tests for the standalone SMS dataset evaluation helpers."""

import io
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_sms_detector import evaluate_rows, print_report, read_dataset


class SmsEvaluationTests(unittest.TestCase):
    def test_confusion_counts_use_spam_as_positive_and_threshold_30(self):
        result = evaluate_rows(
            [
                ("spam", "Reply with your OTP to verify your account now."),
                ("ham", "Hi, I will meet you at the library at 5 pm."),
                ("ham", "Track your parcel here: https://bit.ly/parcel123"),
                ("spam", "Congratulations! You have won a lottery prize. Claim your reward now."),
            ]
        )

        self.assertEqual((result.tp, result.tn, result.fp, result.fn), (1, 1, 1, 1))
        self.assertEqual(result.accuracy, 0.5)
        self.assertEqual(result.precision, 0.5)
        self.assertEqual(result.recall, 0.5)
        self.assertEqual(result.f1_score, 0.5)
        self.assertEqual(result.false_positive_rate, 0.5)

    def test_metrics_are_zero_when_denominators_are_zero(self):
        result = evaluate_rows([])

        self.assertEqual(result.evaluated, 0)
        self.assertEqual(result.accuracy, 0)
        self.assertEqual(result.precision, 0)
        self.assertEqual(result.recall, 0)
        self.assertEqual(result.f1_score, 0)
        self.assertEqual(result.false_positive_rate, 0)

    def test_dataset_reader_skips_malformed_rows_and_invalid_messages(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "messages.tsv"
            path.write_text(
                "ham\tHello there\nspam\t\nunknown\tbad label\nmissing separator\n",
                encoding="utf-8",
            )
            warnings = io.StringIO()
            result = read_dataset(path, warnings)

        self.assertEqual((result.tn, result.tp), (1, 0))
        self.assertEqual(result.skipped_rows, 2)
        self.assertEqual(result.skipped_messages, 1)
        self.assertIn("row 3", warnings.getvalue())
        self.assertIn("row 4", warnings.getvalue())

    def test_report_includes_threshold_confusion_matrix_and_metrics(self):
        result = evaluate_rows([("ham", "Hello there")])
        output = io.StringIO()

        print_report(result, output)

        report = output.getvalue()
        self.assertIn("risk_score >= 30", report)
        self.assertIn("Confusion matrix (spam is positive)", report)
        self.assertIn("False-positive rate", report)


if __name__ == "__main__":
    unittest.main()
