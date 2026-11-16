import numpy as np

from forecasting.models.boosting import (
    feature_gains,
    fit_regressor,
    load_regressor,
    model_filename,
    predict,
    save_regressor,
)


def _sample():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(60, 4))
    features[0, 1] = np.nan
    target = features[:, 0] * 2.0 + rng.normal(0, 0.1, 60)
    return features, target


def _params(model_type: str) -> dict:
    if model_type == "xgboost":
        return {
            "n_estimators": 15,
            "max_depth": 2,
            "learning_rate": 0.1,
            "min_child_weight": 1.0,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_lambda": 1.0,
            "gamma": 0.0,
        }
    return {
        "n_estimators": 15,
        "learning_rate": 0.1,
        "num_leaves": 8,
        "min_child_samples": 5,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_lambda": 1.0,
    }


def test_saved_model_matches_in_memory_predictions(tmp_path):
    features, target = _sample()
    for model_type in ("xgboost", "lightgbm"):
        model = fit_regressor(
            model_type,
            _params(model_type),
            seed=0,
            features=features,
            target=target,
        )
        path = tmp_path / model_type / model_filename(model_type)
        save_regressor(model, model_type, path)
        loaded = load_regressor(model_type, path)
        np.testing.assert_allclose(
            predict(model, features),
            predict(loaded, features),
            rtol=1e-5,
            atol=1e-5,
        )
        gains = feature_gains(model, model_type, features.shape[1])
        assert gains.shape == (features.shape[1],)
        assert np.isfinite(gains).all()
