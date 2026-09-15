"""One prepared-table schema, with explicit observed-data separation."""
import numpy as np
import pandas as pd

FEATURE_KEYS = (
    "mz1", "mz2", "ptz1", "ptz2", "pt4l", "pt_z1_l1", "pt_z1_l2",
    "pt_z2_l1", "pt_z2_l2", "eta_z1_l1", "eta_z1_l2", "eta_z2_l1", "eta_z2_l2",
    "theta1", "theta2", "Theta", "Phi", "Phi1", "dphi_z1", "dphi_z2",
    "dR_z1", "dR_z2", "dR_02", "dR_03", "dR_12", "dR_13",
    "scalar_pt_sum_z1", "scalar_pt_sum_z2", "met", "met_phi", "jet_n"
)

def to_frame(reconstructed, sample, role, source_id=None):
    frame = pd.DataFrame({key: reconstructed[key] for key in FEATURE_KEYS + ("mass",)})
    frame["weight"] = reconstructed["w"]
    frame["sample"] = sample
    if source_id is not None:
        frame["source_id"] = source_id
    frame["role"] = role
    frame["label"] = {"signal": 1, "background": 0, "data": -1}[role]
    return frame

def validate_frame(frame, features):
    unknown = set(features) - set(FEATURE_KEYS)
    if unknown:
        raise ValueError(f"Unrecognized/disallowed ML features: {sorted(unknown)}")
    required = set(features) | {"event_id", "sample", "role", "label", "weight", "mass"}
    if required - set(frame):
        raise ValueError(f"Missing prepared columns: {sorted(required-set(frame))}")
    if frame.empty or frame.event_id.isna().any() or frame.event_id.duplicated().any():
        raise ValueError("Prepared events must be nonempty with unique event_id values")
    if not frame.role.isin(["signal", "background", "data"]).all():
        raise ValueError("Unknown sample role")
    expected = frame.role.map({"signal": 1, "background": 0, "data": -1})
    if not (expected == frame.label).all():
        raise ValueError("Role/label mismatch; observed data must have label -1")
    if not np.isfinite(frame[list(features)+["weight", "mass"]].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite features, mass or weights; inspect the prepared dataset")
    if not (frame.loc[frame.role == "data", "weight"] == 1).all():
        raise ValueError("Observed data must have unit event weights")
    return frame
