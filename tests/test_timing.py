import pandas as pd

from strategy.timing import latest_ai_assessment


def test_latest_assessment_requires_signal_and_probability():
    signals = pd.DataFrame({"AI Probability": [0.65], "Signal": [0]})
    assessment = latest_ai_assessment(signals)
    assert assessment["status"] == "Wait for confirmation"
