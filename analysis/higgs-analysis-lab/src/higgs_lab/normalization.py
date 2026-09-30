"""Auditable checks of luminosity and Monte Carlo event normalization."""
from __future__ import annotations

import re
from dataclasses import replace

import numpy as np
import pandas as pd

from .provenance import new_output, write_json
from .selection import WEIGHT_BRANCHES


CATALOGUE_FIELDS = {
    "xsec": "cross_section_pb",
    "filteff": "genFiltEff",
    "kfac": "kFactor",
    "sum_of_weights": "sumOfWeights",
}


def _dataset_id(url):
    match = re.search(r"(?:^|_)mc_(\d+)\.", str(url))
    if not match:
        raise ValueError(f"Cannot extract MC dataset ID from {url}")
    return match.group(1)


def _relative_difference(value, reference):
    scale = max(abs(float(reference)), np.finfo(float).tiny)
    return abs(float(value) - float(reference)) / scale


def _audit_file(url, sample, role, luminosity_fb, metadata):
    import uproot

    did = _dataset_id(url)
    with uproot.open(url) as root_file:
        tree = root_file["analysis"]
        missing = set(WEIGHT_BRANCHES + ["sum_of_weights"]) - set(tree.keys())
        if missing:
            raise ValueError(f"Missing normalization branches in {url}: {sorted(missing)}")
        arrays = tree.arrays(WEIGHT_BRANCHES + ["sum_of_weights"], library="np")
    if not len(arrays["mcWeight"]):
        raise ValueError(f"Empty MC ROOT tree: {url}")

    row = {
        "sample": sample,
        "role": role,
        "dataset_id": did,
        "source": str(url),
        "skim_entries": len(arrays["mcWeight"]),
    }
    all_constant = True
    catalogue_match = True
    for branch, field in CATALOGUE_FIELDS.items():
        values = np.asarray(arrays[branch], dtype=float)
        root_value = float(values[0])
        spread = float(np.max(np.abs(values - root_value)))
        reference = float(metadata[field])
        relative = _relative_difference(root_value, reference)
        row[f"root_{branch}"] = root_value
        row[f"catalogue_{branch}"] = reference
        row[f"relative_difference_{branch}"] = relative
        row[f"constant_{branch}"] = spread <= 1e-6 * max(1.0, abs(root_value))
        all_constant &= row[f"constant_{branch}"]
        if branch == "sum_of_weights":
            catalogue_match &= relative <= 1e-5

    root_effective = (
        row["root_xsec"] * row["root_filteff"] * row["root_kfac"])
    catalogue_effective = (
        row["catalogue_xsec"]
        * row["catalogue_filteff"]
        * row["catalogue_kfac"])
    effective_relative = _relative_difference(root_effective, catalogue_effective)
    catalogue_match &= effective_relative <= 5e-3
    row.update({
        "root_effective_cross_section_pb": root_effective,
        "catalogue_effective_cross_section_pb": catalogue_effective,
        "relative_difference_effective_cross_section": effective_relative,
    })

    factors = [np.asarray(arrays[name], dtype=float) for name in WEIGHT_BRANCHES]
    signed_factor = np.prod(factors, axis=0)
    absolute_factor = np.prod([np.abs(value) for value in factors], axis=0)
    signed_sum = float(signed_factor.sum())
    absolute_sum = float(absolute_factor.sum())
    normalization = luminosity_fb * 1000.0 / row["root_sum_of_weights"]
    row.update({
        "negative_mcweight_fraction": float(np.mean(factors[3] < 0)),
        "signed_skim_yield": normalization * signed_sum,
        "absolute_skim_yield": normalization * absolute_sum,
        "absolute_to_signed_yield_ratio": (
            absolute_sum / signed_sum if signed_sum != 0 else np.nan),
        "metadata_constant_within_file": bool(all_constant),
        "catalogue_match": bool(catalogue_match),
    })
    return row


def audit_normalization(config, output):
    """Compare ROOT normalization branches with ATLAS catalogue metadata."""
    import atlasopenmagic as atom
    from .datasets import resolve_samples

    output = new_output(output)
    atom.set_release(config.data.release)
    resolved = resolve_samples(config)
    rows = []
    failures = []
    urls = []
    for sample, sample_urls in resolved:
        urls.extend(sample_urls)
        if sample["role"] == "data":
            continue
        for url in sample_urls:
            did = _dataset_id(url)
            print(f"Auditing MC dataset {did}", flush=True)
            try:
                rows.append(_audit_file(
                    url, sample["name"], sample["role"],
                    config.data.luminosity_fb, atom.get_metadata(did)))
            except (OSError, TypeError) as exc:
                failures.append({
                    "dataset_id": did, "source": str(url), "error": str(exc)})
                print(f"Warning: could not read {did}: {exc}", flush=True)
            if rows:
                pd.DataFrame(rows).sort_values(
                    ["role", "dataset_id"]).to_csv(
                        output / "normalization_audit.csv", index=False)

    if not rows:
        raise OSError("No MC ROOT files could be audited")

    table = pd.DataFrame(rows).sort_values(["role", "dataset_id"])
    table.to_csv(output / "normalization_audit.csv", index=False)
    duplicate_urls = sorted({url for url in urls if urls.count(url) > 1})
    problematic = table.loc[
        (~table.catalogue_match)
        | (~table.metadata_constant_within_file)
        | (table.absolute_to_signed_yield_ratio > 1.01),
        "dataset_id",
    ].astype(str).tolist()
    summary = {
        "release": config.data.release,
        "skim": config.data.skim,
        "luminosity_fb_inverse": config.data.luminosity_fb,
        "processed_fraction": config.data.fraction,
        "configured_weight_mode": config.data.weight_mode,
        "weight_formula": (
            "L[fbinv] * 1000[pb/fb] / sumOfWeights * xsec[pb] * "
            "genFiltEff * kFactor * mcWeight * pileup/electron/muon/trigger scale factors"),
        "luminosity_cross_section_unit_conversion": (
            "1000 converts fb^-1 times pb to event yield"),
        "mc_files": int(len(table)),
        "duplicate_urls": duplicate_urls,
        "read_failures": failures,
        "audit_complete": not failures,
        "all_catalogue_values_match": bool(table.catalogue_match.all()),
        "all_background_effective_cross_sections_match_catalogue": bool(
            table.loc[table.role == "background", "catalogue_match"].all()),
        "all_metadata_constant_within_files": bool(
            table.metadata_constant_within_file.all()),
        "files_with_negative_generator_weights": table.loc[
            table.negative_mcweight_fraction > 0,
            "dataset_id"].astype(str).tolist(),
        "files_with_more_than_one_percent_absolute_weight_inflation": table.loc[
            table.absolute_to_signed_yield_ratio > 1.01,
            "dataset_id"].astype(str).tolist(),
        "problematic_or_review_required_dataset_ids": problematic,
        "interpretation": [
            "The physical product cross section times filter efficiency times k-factor, plus sum of weights, is compared with the ATLAS catalogue.",
            "Individual catalogue fields can use a different factorization convention from the ROOT ntuple, especially for Higgs branching fractions.",
            "No duplicate URL should appear in the resolved sample list.",
            "An absolute-to-signed ratio above one is caused by discarding negative generator-weight signs.",
            "legacy_absolute reproduces the original student code but is not automatically a physics-correct NLO convention.",
            "This audit does not establish that every required physics background process is present.",
        ],
    }
    write_json(output / "normalization_audit.json", summary)
    return output


def compare_weight_conventions(config, output, fit_range=(105., 140.),
                               signal_window=(118., 130.)):
    """Compare signed and absolute MC yields after the full local selection."""
    import awkward as ak
    from .datasets import resolve_samples, iter_batches
    from .selection import select_events

    low, high = map(float, fit_range)
    signal_low, signal_high = map(float, signal_window)
    if not low < signal_low < signal_high < high:
        raise ValueError("Require fit_low < signal_low < signal_high < fit_high")
    output = new_output(output)
    signed_config = replace(
        config, data=replace(config.data, weight_mode="signed"))
    regions = {
        "all_selected": (-np.inf, np.inf),
        "fit_range": (low, high),
        "lower_sideband": (low, signal_low),
        "signal_window": (signal_low, signal_high),
        "upper_sideband": (signal_high, high),
    }
    accumulators = {}
    for sample, urls in resolve_samples(config):
        for url in urls:
            print(f"Comparing weights for {sample['name']}: {url}", flush=True)
            for events in iter_batches(url, config, sample["role"]):
                selected, _ = select_events(events, signed_config, sample["role"])
                masses = ak.to_numpy(selected["mass"])
                signed = ak.to_numpy(selected["totalWeight"])
                absolute = (
                    np.ones(len(selected)) if sample["role"] == "data"
                    else np.abs(signed))
                for region, (region_low, region_high) in regions.items():
                    mask = (masses >= region_low) & (masses < region_high)
                    key = (sample["name"], sample["role"], region)
                    target = accumulators.setdefault(key, {
                        "events": 0, "signed_yield": 0., "signed_sumw2": 0.,
                        "absolute_yield": 0., "absolute_sumw2": 0.})
                    target["events"] += int(mask.sum())
                    target["signed_yield"] += float(signed[mask].sum())
                    target["signed_sumw2"] += float(np.square(signed[mask]).sum())
                    target["absolute_yield"] += float(absolute[mask].sum())
                    target["absolute_sumw2"] += float(np.square(absolute[mask]).sum())

    rows = []
    for (sample, role, region), values in accumulators.items():
        signed_yield = values["signed_yield"]
        absolute_yield = values["absolute_yield"]
        rows.append({
            "sample": sample, "role": role, "region": region, **values,
            "absolute_minus_signed": absolute_yield - signed_yield,
            "absolute_to_signed_ratio": (
                absolute_yield / signed_yield if signed_yield != 0 else np.nan),
            "signed_stat_error": np.sqrt(values["signed_sumw2"]),
            "absolute_stat_error": np.sqrt(values["absolute_sumw2"]),
        })
    detail = pd.DataFrame(rows)
    detail.to_csv(output / "weight_convention_by_sample.csv", index=False)

    aggregate = (detail.groupby(["role", "region"], as_index=False)
                 .agg(events=("events", "sum"),
                      signed_yield=("signed_yield", "sum"),
                      signed_sumw2=("signed_sumw2", "sum"),
                      absolute_yield=("absolute_yield", "sum"),
                      absolute_sumw2=("absolute_sumw2", "sum")))
    aggregate["absolute_minus_signed"] = (
        aggregate.absolute_yield - aggregate.signed_yield)
    aggregate["absolute_to_signed_ratio"] = (
        aggregate.absolute_yield / aggregate.signed_yield)
    aggregate["signed_stat_error"] = np.sqrt(aggregate.signed_sumw2)
    aggregate["absolute_stat_error"] = np.sqrt(aggregate.absolute_sumw2)
    aggregate.to_csv(output / "weight_convention_summary.csv", index=False)
    write_json(output / "weight_convention_method.json", {
        "luminosity_fb_inverse": config.data.luminosity_fb,
        "fit_range_gev": [low, high],
        "signal_window_gev": [signal_low, signal_high],
        "selection": "Identical four-lepton selection in both conventions",
        "signed": "Generator-weight signs retained; denominator is signed sumOfWeights",
        "legacy_absolute": "Absolute value applied to every multiplicative weight branch",
        "note": "This is a normalization diagnostic and does not retrain classifiers.",
    })
    return output
