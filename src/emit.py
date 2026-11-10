"""Write and validate the submission CSV.

Format contract (SUBMISSION.md / evaluation/validate_submission.py):
  * header ``frame,label``
  * exactly 199319 data rows, frame = 0..N-1 in order (integer dtype)
  * label in {H, A, D, X} (stripped/upper-cased on read)
  * UTF-8, comma separated, no index column
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LABEL_STR = np.array(["H", "A", "D", "X"])


def labels_to_csv(labels: np.ndarray, path: str, n_expected: int | None = None) -> str:
    if n_expected is not None and len(labels) != n_expected:
        raise ValueError(f"expected {n_expected} rows, got {len(labels)}")
    codes = np.asarray(labels)
    if codes.dtype.kind not in "iu":
        raise TypeError("labels must be integer codes 0..3")
    if codes.min() < 0 or codes.max() > 3:
        raise ValueError("label codes out of range")
    str_labels = LABEL_STR[codes]
    df = pd.DataFrame({"frame": np.arange(len(codes), dtype=np.int64), "label": str_labels})
    df.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    return path


def validate(path: str, n_expected: int) -> None:
    """Mirror of the official validator's checks (raises ValueError on fail)."""
    df = pd.read_csv(path)
    cols = {str(c).strip().lower() for c in df.columns}
    if not {"frame", "label"} <= cols:
        raise ValueError(f"missing frame/label columns: {list(df.columns)}")
    df.columns = [str(c).strip().lower() for c in df.columns]
    if len(df) != n_expected:
        raise ValueError(f"expected {n_expected} rows, got {len(df)}")
    if df["frame"].dtype.kind == "f":
        raise ValueError("frame column parsed as float")
    frames = df["frame"].to_numpy()
    if not np.array_equal(frames, np.arange(n_expected)):
        raise ValueError("frame column is not 0..N-1 in order")
    labels = df["label"].astype(str).str.strip().str.upper().to_numpy()
    bad = sorted(set(labels.tolist()) - {"H", "A", "D", "X"})
    if bad:
        raise ValueError(f"invalid labels: {bad}")
    if df.isna().any().any():
        raise ValueError("blank cells in submission")
