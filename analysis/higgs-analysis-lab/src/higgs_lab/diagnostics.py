"""Prepared-data plotting orchestration."""
import json
from pathlib import Path
import pandas as pd

from .features import FEATURE_KEYS, validate_frame
from .plots import save_feature_plots
from .provenance import new_output, sha256, write_json

def create_diagnostics(prepared, output):
    prepared = Path(prepared)
    manifest = json.loads((prepared/"manifest.json").read_text())
    csv_path = prepared/"features.csv"
    if sha256(csv_path) != manifest.get("features_sha256"):
        raise ValueError("Prepared feature checksum mismatch")
    frame = pd.read_csv(csv_path)
    validate_frame(frame, FEATURE_KEYS)
    output = new_output(output)
    save_feature_plots(frame, output, FEATURE_KEYS)
    write_json(output/"diagnostics.json", {
        "features": list(FEATURE_KEYS), "plots": len(FEATURE_KEYS) + 1,
        "note": "MC histograms use physical event weights; correlation uses signal events."
    })
    return output
