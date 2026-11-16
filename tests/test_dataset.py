import pandas as pd

from forecasting.data.dataset import OPSD_SHA256, calendar_gaps, load_daily_load
from forecasting.data.download import sha256_bytes, verify_opsd_bytes


def test_vendored_opsd_matches_the_pinned_checksum(opsd_csv):
    payload = opsd_csv.read_bytes()
    verify_opsd_bytes(payload)
    assert sha256_bytes(payload) == OPSD_SHA256


def test_checksum_rejects_a_modified_file():
    try:
        verify_opsd_bytes(b"not the opsd file")
    except ValueError as exc:
        assert "checksum mismatch" in str(exc)
    else:
        raise AssertionError("expected a checksum error")


def test_loader_returns_a_complete_daily_series(opsd_csv):
    frame = load_daily_load(opsd_csv)
    assert list(frame.columns) == ["date", "consumption", "wind", "solar"]
    assert len(frame) == 4383
    assert frame["date"].iloc[0] == pd.Timestamp("2006-01-01")
    assert frame["date"].iloc[-1] == pd.Timestamp("2017-12-31")
    assert calendar_gaps(frame).empty
    assert (frame["consumption"] > 0).all()
    assert frame["wind"].isna().any()
    assert frame["solar"].isna().any()
    assert frame["date"].is_monotonic_increasing
