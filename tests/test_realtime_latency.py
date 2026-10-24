import csv
from contextlib import redirect_stdout
import json
from io import StringIO
import tempfile
import unittest
from pathlib import Path

from scripts import simulate_realtime


class RealtimeLatencyTests(unittest.TestCase):
    def test_print_summary_includes_suspicious_transaction_count(self):
        output = StringIO()
        with redirect_stdout(output):
            simulate_realtime.print_summary(
                [0.01],
                [0.002],
                [0.003],
                total_rows=2,
                error_rows=1,
                total_time=0.01,
                fraud_transactions=1,
            )

        self.assertIn("Số giao dịch nghi ngờ gian lận: 1 (100.00%)", output.getvalue())

    def test_latency_distribution_plot_is_created(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib is not installed")

        records = [
            {"status": "success", "total_ms": 10.0, "is_cold_start": False},
            {"status": "success", "total_ms": 25.0, "is_cold_start": False},
            {"status": "success", "total_ms": 110.0, "is_cold_start": False},
        ]
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "latency_distribution.png"
            result = simulate_realtime._write_latency_distribution_plot(
                records, output_path
            )
            self.assertEqual(result, output_path)
            self.assertGreater(output_path.stat().st_size, 0)

    def test_latency_stats_percentiles_and_cold_start_filter(self):
        records = [
            {"status": "success", "total_ms": 10.0, "is_cold_start": False},
            {"status": "success", "total_ms": 20.0, "is_cold_start": False},
            {"status": "success", "total_ms": 1000.0, "is_cold_start": True},
        ]

        stats = simulate_realtime._latency_stats(records, "total_ms")

        self.assertEqual(stats["min"], 10.0)
        self.assertEqual(stats["max"], 20.0)
        self.assertEqual(stats["p95"], 19.5)
        self.assertEqual(
            simulate_realtime._latency_stats(
                records, "total_ms", include_cold_start=True
            )["max"],
            1000.0,
        )

    def test_benchmark_outputs_write_required_metrics_and_summary(self):
        records = [
            {
                "transaction_index": 1,
                "sender_account": "sender",
                "receiver_account": "receiver",
                "fraud_probability": 0.9,
                "is_fraud": True,
                "data_processing_ms": 1.0,
                "graphsage_inference_ms": 2.0,
                "xai_subgraph_ms": 3.0,
                "gnnexplainer_ms": 4.0,
                "reason_code_ms": 5.0,
                "total_ms": 15.0,
                "explanation_generated": True,
                "status": "success",
                "error_type": "",
                "error_message": "",
                "is_cold_start": False,
            },
            {
                "transaction_index": 2,
                "sender_account": "bad",
                "receiver_account": "row",
                "fraud_probability": "",
                "is_fraud": "",
                "data_processing_ms": "",
                "graphsage_inference_ms": "",
                "xai_subgraph_ms": "",
                "gnnexplainer_ms": "",
                "reason_code_ms": "",
                "total_ms": "",
                "explanation_generated": False,
                "status": "error",
                "error_type": "ValueError",
                "error_message": "invalid row",
                "is_cold_start": False,
            },
        ]

        with tempfile.TemporaryDirectory() as directory:
            metrics_path = Path(directory) / "latency.csv"
            summary_path = Path(directory) / "summary.json"
            simulate_realtime._write_benchmark_outputs(
                records,
                total_rows=2,
                error_rows=1,
                total_time_ms=20.0,
                metrics_output=metrics_path,
                summary_output=summary_path,
                enable_xai=True,
                xai_epochs=2,
                xai_num_hops=2,
                warmup_runs=1,
                include_cold_start=False,
            )

            with metrics_path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(len(rows), 2)
        self.assertIn("gnnexplainer_ms", rows[0])
        self.assertEqual(summary["error_transactions"], 1)
        self.assertEqual(summary["xai_transactions"], 1)
        self.assertEqual(summary["warmup_runs"], 1)
        self.assertEqual(summary["statistics"]["all"]["total_ms"]["p95"], 15.0)


if __name__ == "__main__":
    unittest.main()
