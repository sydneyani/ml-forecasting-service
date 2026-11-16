import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_quotes_the_holdout_numbers():
    report = json.loads((ROOT / "reports" / "backtest_results.json").read_text())
    champion = report["champion_model_type"]
    mae = report["models"][champion]["test"]["mae"]
    readme = (ROOT / "README.md").read_text()
    walkthrough = (ROOT / "docs" / "WALKTHROUGH.md").read_text()
    card = (ROOT / "docs" / "MODEL_CARD.md").read_text()
    rendered = f"{mae:.2f}"
    assert rendered in readme
    assert rendered in walkthrough
    assert rendered in card
    assert report["model_version"] in readme
    assert f"{report['mae_ratio_vs_seasonal_naive_lag_364']:.3f}" in readme
