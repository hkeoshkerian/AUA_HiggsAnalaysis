"""Event cuts and weight calculation, independent of I/O and plotting."""
import numpy as np
import awkward as ak
from .reconstruction import calc_mass

KINEMATIC_BRANCHES = [
    "lep_pt", "lep_eta", "lep_phi", "lep_e", "lep_charge", "lep_type",
    "trigE", "trigM", "lep_isTrigMatched", "lep_isLooseID",
    "lep_isMediumID", "lep_isLooseIso", "lep_n", "jet_n", "met", "met_phi"
]
WEIGHT_BRANCHES = ["filteff", "kfac", "xsec", "mcWeight", "ScaleFactor_PILEUP",
                   "ScaleFactor_ELE", "ScaleFactor_MUON", "ScaleFactor_LepTRIGGER"]

def calculate_weights(events, luminosity_fb, mode="legacy_absolute"):
    if mode not in {"legacy_absolute", "signed"}:
        raise ValueError("Unknown weight mode")
    denom = ak.to_numpy(events["sum_of_weights"])
    if np.any(~np.isfinite(denom)) or np.any(denom == 0):
        raise ValueError("Invalid sum_of_weights")
    weights = luminosity_fb * 1000 / denom
    for name in WEIGHT_BRANCHES:
        values = ak.to_numpy(events[name])
        weights = weights * (np.abs(values) if mode == "legacy_absolute" else values)
    if not np.isfinite(weights).all():
        raise ValueError("Nonfinite event weights")
    return weights

def select_events(events, config, role):
    """Return selected events and raw cutflow. Input energies are GeV.

    Enforce the exactly-four/sorted-pT contract assumed by the source skim.
    Thresholds are strict > comparisons, matching the uploaded code.
    """
    if role not in {"data", "signal", "background"}:
        raise ValueError("Invalid sample role")
    required = KINEMATIC_BRANCHES + ([] if role == "data" else WEIGHT_BRANCHES + ["sum_of_weights"])
    missing = set(required) - set(events.fields)
    if missing:
        raise ValueError(f"Missing ROOT branches: {sorted(missing)}")
    rows = []
    def record(stage):
        rows.append({"stage": stage, "events": len(current)})
    current = events
    record("input")
    current = current[(current.lep_n == 4) & (ak.num(current.lep_pt, axis=1) == 4)]
    record("exactly_four_leptons")
    for name in [n for n in KINEMATIC_BRANCHES if n.startswith("lep_") and n != "lep_n"]:
        if ak.any(ak.num(current[name], axis=1) != 4):
            raise ValueError(f"Inconsistent lepton array length: {name}")
    if config.selection.sort_leptons_by_pt:
        # Optional safer mode: sort every per-lepton field with identical indices.
        order = ak.argsort(current.lep_pt, axis=1, ascending=False)
        lepton_fields = [name for name in KINEMATIC_BRANCHES
                          if name.startswith("lep_") and name != "lep_n"]
        for name in lepton_fields:
            current = ak.with_field(current, current[name][order], name)
    current = current[current.trigE | current.trigM]
    record("trigger")
    current = current[ak.sum(current.lep_isTrigMatched, axis=1) >= 1]
    record("trigger_match")
    for index, threshold in enumerate(config.selection.pt_min_gev):
        current = current[current.lep_pt[:, index] > threshold]
        record(f"pt_{index+1}_gt_{threshold:g}_GeV")
    good = (((current.lep_type == 11) & current.lep_isLooseID & current.lep_isLooseIso)
            | ((current.lep_type == 13) & current.lep_isMediumID & current.lep_isLooseIso))
    current = current[ak.sum(good, axis=1) == 4]
    record("id_and_isolation")
    types = ak.sum(current.lep_type, axis=1)
    current = current[(types == 44) | (types == 48) | (types == 52)]
    record("flavour")
    current = current[ak.sum(current.lep_charge, axis=1) == 0]
    record("zero_total_charge")
    current = ak.with_field(current, calc_mass(current.lep_pt, current.lep_eta, current.lep_phi, current.lep_e), "mass")
    weights = np.ones(len(current)) if role == "data" else calculate_weights(current, config.data.luminosity_fb, config.data.weight_mode)
    current = ak.with_field(current, weights, "totalWeight")
    return current, rows
