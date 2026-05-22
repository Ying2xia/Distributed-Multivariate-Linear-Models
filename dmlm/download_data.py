"""Download helpers for the real-data applications."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve
import zipfile

import pandas as pd


BEIJING_UCI_URL = "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip"
SARCOS_TRAIN_URL = "https://gaussianprocess.org/gpml/data/sarcos_inv.mat"
SARCOS_TEST_URL = "https://gaussianprocess.org/gpml/data/sarcos_inv_test.mat"


def _download(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    print(f"Downloading {url}")
    urlretrieve(url, destination)
    return destination


def download_beijing(output_dir: str | Path = "data/beijing") -> Path:
    """Download and extract the Beijing Multi-Site Air Quality dataset."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    zip_path = output / "beijing_multi_site_air_quality.zip"
    marker = output / ".download_complete"
    if marker.exists() and list(output.rglob("PRSA_Data_*.csv")):
        return output

    _download(BEIJING_UCI_URL, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(output)
    for nested_zip in output.rglob("*.zip"):
        if nested_zip == zip_path:
            continue
        with zipfile.ZipFile(nested_zip) as archive:
            archive.extractall(output)
    marker.write_text("ok\n")
    return output


def _load_mat_array(path: Path, preferred_key: str):
    try:
        from scipy.io import loadmat
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "SARCOS is distributed as MATLAB .mat files. Install scipy first, e.g. "
            "`python3 -m pip install scipy`, then rerun with --download."
        ) from exc

    data = loadmat(path)
    if preferred_key in data:
        return data[preferred_key]
    candidates = [value for key, value in data.items() if not key.startswith("__") and getattr(value, "ndim", 0) == 2]
    if not candidates:
        raise ValueError(f"No 2D numeric array found in {path}")
    return candidates[0]


def download_sarcos(output_dir: str | Path = "data/sarcos") -> tuple[Path, Path]:
    """Download the SARCOS inverse dynamics data and convert it to CSV."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    train_mat = _download(SARCOS_TRAIN_URL, output / "sarcos_inv.mat")
    test_mat = _download(SARCOS_TEST_URL, output / "sarcos_inv_test.mat")
    train_csv = output / "sarcos_inv.csv"
    test_csv = output / "sarcos_inv_test.csv"

    if not train_csv.exists():
        train = _load_mat_array(train_mat, "sarcos_inv")
        pd.DataFrame(train).to_csv(train_csv, header=False, index=False)
    if not test_csv.exists():
        test = _load_mat_array(test_mat, "sarcos_inv_test")
        pd.DataFrame(test).to_csv(test_csv, header=False, index=False)

    return train_csv, test_csv
