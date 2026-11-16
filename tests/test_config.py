from pathlib import Path

from forecasting.config import load_config, resolve_paths

ROOT = Path(__file__).resolve().parents[1]


def test_repository_configs_validate(monkeypatch):
    monkeypatch.chdir(ROOT)
    train = load_config(ROOT / "configs" / "train.yaml")
    ci = load_config(ROOT / "configs" / "ci.yaml")
    assert train.fast is False
    assert train.tracking.enabled is True
    assert {item.model_type for item in train.search} == {"xgboost", "lightgbm"}
    assert ci.fast is True
    assert ci.search[0].n_trials <= 3
    resolved = resolve_paths(train, ROOT)
    assert resolved.data.csv_path.exists()
    assert resolved.output.model_dir.name == "champion"
