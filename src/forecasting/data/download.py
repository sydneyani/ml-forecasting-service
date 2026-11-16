"""Download the pinned OPSD daily file and check its SHA-256."""

import argparse
import hashlib
import urllib.error
import urllib.request
from pathlib import Path

from forecasting.data.dataset import OPSD_DAILY_URL, OPSD_SHA256


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def verify_opsd_bytes(payload: bytes) -> None:
    digest = sha256_bytes(payload)
    if digest != OPSD_SHA256:
        raise ValueError(f"OPSD daily file checksum mismatch: {digest}")


def download_opsd(dest: Path, url: str = OPSD_DAILY_URL, timeout: int = 60) -> Path:
    """Fetch the pinned file and replace `dest` only after the checksum matches."""
    request = urllib.request.Request(url, headers={"User-Agent": "ml-forecasting-service/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(f"failed to download the OPSD daily series: {exc}") from exc
    verify_opsd_bytes(payload)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(dest.suffix + ".partial")
    temporary.write_bytes(payload)
    temporary.replace(dest)
    return dest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download the pinned OPSD Germany daily file.")
    parser.add_argument("--dest", type=Path, default=Path("data/raw/opsd_germany_daily.csv"))
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download even when dest already matches.",
    )
    args = parser.parse_args(argv)
    dest = args.dest
    if dest.exists() and not args.force:
        try:
            verify_opsd_bytes(dest.read_bytes())
        except ValueError:
            pass
        else:
            print(f"{dest} already matches the pinned OPSD checksum")
            return
    download_opsd(dest)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
