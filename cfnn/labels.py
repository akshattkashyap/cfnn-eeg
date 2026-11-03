"""Build a separate DEAP copy with valence/arousal taken from keyed metadata."""
from __future__ import annotations

import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

LABELS = ("valence", "arousal", "dominance", "liking")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_ratings(path):
    import xlrd

    book = xlrd.open_workbook(str(path))
    try:
        sheet = book.sheet_by_index(0)
        headers = [str(value).strip().lower() for value in sheet.row_values(0)]
        required = ("participant_id", "experiment_id", *LABELS)
        if any(headers.count(name) != 1 for name in required):
            raise ValueError("Metadata must have unique participant, experiment and rating columns")
        columns = {name: headers.index(name) for name in required}
        ratings = {}
        for row in range(1, sheet.nrows):
            if not any(str(v).strip() for v in sheet.row_values(row)):
                continue
            ids = [float(sheet.cell_value(row, columns[name]))
                   for name in ("participant_id", "experiment_id")]
            if not all(np.isfinite(v) and v.is_integer() for v in ids):
                raise ValueError(f"Invalid metadata IDs on spreadsheet row {row + 1}")
            key = tuple(int(v) for v in ids)
            if key in ratings:
                raise ValueError(f"Duplicate metadata key {key}")
            values = np.array([sheet.cell_value(row, columns[name]) for name in LABELS], float)
            if not np.isfinite(values).all() or np.any((values < 1) | (values > 9)):
                raise ValueError(f"Invalid metadata ratings for {key}")
            ratings[key] = values
        expected = {(s, t) for s in range(1, 33) for t in range(1, 41)}
        if set(ratings) != expected:
            raise ValueError("Metadata must cover exactly 32 participants and 40 videos each")
        return ratings
    finally:
        book.release_resources()


def corrected_labels(actual, expected):
    actual = np.asarray(actual)
    if actual.shape != (40, 4) or not np.isfinite(actual).all():
        raise ValueError("Expected 40 finite rows and four label columns")
    if not np.issubdtype(actual.dtype, np.floating):
        raise ValueError("Expected floating-point DEAP ratings")
    if not np.allclose(actual[:, 2:], expected[:, 2:], rtol=0, atol=1e-6):
        raise ValueError("Dominance/liking mismatch: participant/video alignment is not verified")
    corrected = actual.copy()
    corrected[:, :2] = expected[:, :2]
    return corrected


def repair_labels(source, metadata, destination):
    source, metadata, destination = map(lambda p: Path(p).resolve(),
                                         (source, metadata, destination))
    if destination == source or destination in source.parents or source in destination.parents:
        raise ValueError("Source and destination must be separate directory trees")
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}; choose a new directory")
    references = read_ratings(metadata)
    metadata_hash = sha256(metadata)
    files = [source / f"s{s:02d}.dat" for s in range(1, 33)]
    if not all(f.is_file() for f in files):
        raise ValueError("All 32 source subject files are required")
    subjects, changes = [], []
    high_before, high_after = np.zeros(2, int), np.zeros(2, int)
    # Validate every subject before creating the output directory.
    for sid, path in enumerate(files, 1):
        with path.open("rb") as stream:
            obj = pickle.load(stream, encoding="latin1")
        if np.shape(obj["data"]) != (40, 40, 8064):
            raise ValueError(f"Unexpected signal shape in {path.name}")
        actual = np.asarray(obj["labels"])
        expected = np.stack([references[(sid, t)] for t in range(1, 41)])
        corrected = corrected_labels(actual, expected)
        mismatch = ~np.isclose(actual[:, :2], corrected[:, :2], rtol=0, atol=1e-6)
        for trial, column in np.argwhere(mismatch):
            changes.append({"participant_id": sid, "experiment_id": int(trial + 1),
                            "target": LABELS[column], "before": float(actual[trial, column]),
                            "after": float(corrected[trial, column])})
        high_before += (actual[:, :2] >= 5).sum(0)
        high_after += (corrected[:, :2] >= 5).sum(0)
        subjects.append({"file": path.name, "source_sha256": sha256(path),
                         "changed_counts": mismatch.sum(0).tolist()})
        del obj
    destination.mkdir(parents=True, exist_ok=False)
    for sid, (path, record) in enumerate(zip(files, subjects), 1):
        if sha256(path) != record["source_sha256"]:
            raise RuntimeError(f"Source changed during repair: {path}")
        with path.open("rb") as stream:
            obj = pickle.load(stream, encoding="latin1")
        expected = np.stack([references[(sid, t)] for t in range(1, 41)])
        labels = corrected_labels(obj["labels"], expected)
        obj["labels"] = labels
        output = destination / path.name
        with output.open("xb") as stream:
            pickle.dump(obj, stream, protocol=pickle.HIGHEST_PROTOCOL)
        # Reopen each serialized output and verify signal values and labels.
        with output.open("rb") as stream:
            check = pickle.load(stream, encoding="latin1")
        np.testing.assert_array_equal(check["data"], obj["data"])
        np.testing.assert_array_equal(check["labels"], labels)
        record["output_sha256"] = sha256(output)
        del obj, check
        print("Verified", output.name, flush=True)
    if sha256(metadata) != metadata_hash:
        raise RuntimeError("Metadata changed during repair")
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "source_directory": str(source), "output_directory": str(destination),
              "metadata_file": str(metadata), "metadata_sha256": metadata_hash,
              "mapping": "Participant_id and Experiment_id; .dat row = Experiment_id - 1",
              "source_files_modified": False, "threshold_for_counts": 5,
              "high_counts_before": high_before.tolist(), "high_counts_after": high_after.tolist(),
              "changed_rating_counts": [sum(c["target"] == name for c in changes)
                                         for name in LABELS[:2]],
              "subjects": subjects, "changes": changes, "complete": True}
    (destination / "label_repair.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Changed ratings [valence, arousal]:", report["changed_rating_counts"])
    print("High trials after correction:", report["high_counts_after"])
    print("Saved verified dataset:", destination)
    return report
