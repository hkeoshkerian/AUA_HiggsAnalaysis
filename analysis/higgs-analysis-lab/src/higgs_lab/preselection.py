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
    data_settings = manifest.get("settings", {}).get("data", {})
    selection_settings = manifest.get("settings", {}).get("selection", {})
    selection_mode = selection_settings.get("mode", "reference")
    luminosity = float(data_settings.get("luminosity_fb", 36.6))
    mass_range = tuple(selection_settings.get("retained_mass_range_gev", (80.0, 250.0)))
    if selection_mode != "atlas_2017_fiducial": mass_range = (80.0, 250.0)
    save_preselection_mass_plot(frame, output/"m4l_reference_preselection.png",
                                include_data=False, luminosity_fb=luminosity,
                                mass_range=mass_range)
    save_preselection_mass_plot(frame, output/"m4l_data_mc.png",
                                include_data=True, luminosity_fb=luminosity,
                                mass_range=mass_range)
    write_json(output/"preselection_plot.json", {
        "source": str(prepared),
        "events": len(frame),
        "binning_gev": {"minimum": mass_range[0], "maximum": mass_range[1], "width": 2.5},
        "stage": "after event selection and reconstruction; before machine learning"
    })
    return output
