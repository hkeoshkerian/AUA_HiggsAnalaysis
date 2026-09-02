"""Pre-ML plots made directly from a verified prepared cache."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .plots import save_preselection_mass_plot
from .provenance import new_output, sha256, write_json


def plot_preselection(prepared, output):
    prepared = Path(prepared)
    manifest = json.loads((prepared/"manifest.json").read_text())
    csv_path = prepared/"preselection_mass.csv"
    if not csv_path.is_file():
        raise ValueError("Prepared cache predates reference-compatible preselection output; run prepare again")
    if sha256(csv_path) != manifest.get("preselection_mass_sha256"):
        raise ValueError("Prepared preselection-mass checksum mismatch")
    frame = pd.read_csv(csv_path)
    required = {"mass", "weight", "sample", "role"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Missing prepared columns: {sorted(missing)}")
    if frame.empty or not np.isfinite(frame[["mass", "weight"]].to_numpy(float)).all():
        raise ValueError("Prepared mass and weight values must be finite and nonempty")
    if not frame.role.isin(["signal", "background", "data"]).all():
        raise ValueError("Unknown sample role")
    output = new_output(output)
    save_preselection_mass_plot(frame, output/"m4l_reference_preselection.png", include_data=False)
    save_preselection_mass_plot(frame, output/"m4l_data_mc.png", include_data=True)
    write_json(output/"preselection_plot.json", {
        "source": str(prepared),
        "events": len(frame),
        "binning_gev": {"minimum": 80.0, "maximum": 250.0, "width": 2.5},
        "stage": "after event selection and reconstruction; before machine learning"
    })
    return output
