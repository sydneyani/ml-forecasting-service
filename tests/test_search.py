from forecasting.training.search import SearchResult, choose_champion


def _result(model_type: str, cv_mae: float) -> SearchResult:
    return SearchResult(
        model_type=model_type,
        seed=1,
        params={},
        cv_mae=cv_mae,
        cv_mae_std=0.0,
        fold_maes=[cv_mae],
        trials=[],
    )


def test_ties_break_toward_xgboost_and_a_lower_score_wins():
    xgboost = _result("xgboost", 1.0)
    lightgbm = _result("lightgbm", 1.0)
    assert choose_champion([lightgbm, xgboost]).model_type == "xgboost"
    assert choose_champion([xgboost, _result("lightgbm", 0.9)]).model_type == "lightgbm"
