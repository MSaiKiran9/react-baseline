from evaluation.compare import format_comparison_table


def test_format_comparison_table_contains_future_columns() -> None:
    table = format_comparison_table(
        [
            {
                "Model": "qwen3:8b",
                "Dataset": "HotpotQA",
                "System": "ReAct Baseline",
                "n": "20",
                "Accuracy": "0.500",
                "Precision": "0.600",
                "Recall": "0.700",
                "Token F1": "0.400",
                "Sem F1": "N/A",
                "MAR": "N/A",
                "MKG Matches / n": "N/A",
                "Retrieval Failures": "2",
                "Reasoning Failures": "1",
                "Accuracy Improvement": "N/A",
                "Sem F1 Change": "N/A",
                "Average Latency": "4.00",
                "Average Iterations": "2.00",
                "Notes": "baseline",
            }
        ]
    )

    assert "MKG Matches / n" in table
    assert "Sem F1" in table
    assert "Precision" in table
    assert "qwen3:8b" in table
