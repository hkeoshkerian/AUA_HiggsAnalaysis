"""Explicit prepare/run stages. Existing output directories are never overwritten."""
import json
from pathlib import Path
import pandas as pd
from .features import to_frame, validate_frame
from .provenance import environment, new_output, preparation_settings, sha256, write_json

def prepare(config, output):
    import awkward as ak
    from .datasets import resolve_samples, iter_batches
    from .selection import select_events
    from .reconstruction import reconstruct_z1_z2_fast
    output = new_output(output)
    csv_path = output / "features.csv"
    preselection_path = output / "preselection_mass.csv"
    first = True
    first_preselection = True
    cutflows, sources = [], []
    event_number = 0
    for sample, urls in resolve_samples(config):
        print(f"Preparing {sample['name']}",flush=True)
        for url in urls:
            sources.append({"sample":sample["name"],"role":sample["role"],"url":url})
            for events in iter_batches(url,config,sample["role"]):
                selected, cuts = select_events(events,config,sample["role"])
                count = 0
                if len(selected):
                    preselection = pd.DataFrame({
                        "mass": ak.to_numpy(selected["mass"]),
                        "weight": ak.to_numpy(selected["totalWeight"]),
                        "sample": sample["name"],
                        "role": sample["role"],
                    })
                    preselection.to_csv(preselection_path, index=False,
                        mode="w" if first_preselection else "a", header=first_preselection)
                    first_preselection = False
                    rec = reconstruct_z1_z2_fast(selected)
                    frame = to_frame(rec,sample["name"],sample["role"])
                    count = len(frame)
                    if count:
                        frame.insert(0,"event_id",range(event_number,event_number+count))
                        event_number += count
                        frame.to_csv(csv_path,index=False,mode="w" if first else "a",header=first)
                        first = False
                cuts.append({"stage":"two_sfos_pairs","events":count})
                cutflows.extend(dict(sample=sample["name"],**row) for row in cuts)
    if first:
        raise ValueError("No events survived; inspect the input and selections. No valid cache created.")
    pd.DataFrame(cutflows).groupby(["sample","stage"],sort=False,as_index=False).events.sum().to_csv(output/"cutflow.csv",index=False)
    manifest = {"schema_version":1,"settings":preparation_settings(config),"sources":sources,
                "events":event_number,"features_sha256":sha256(csv_path),
                "preselection_mass_sha256":sha256(preselection_path),"environment":environment(),
                "energy_unit":"GeV","event_id_note":"Sequential prepared-row identifier, not a detector event number"}
    write_json(output/"manifest.json",manifest)
    return output

def run(config, prepared, output):
    from .training import train_models
    from .statistics import summarize_mc
    from .plots import save_mass_plot, save_roc_plot, save_detailed_mass_plot
    prepared = Path(prepared)
    manifest = json.loads((prepared/"manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("energy_unit") != "GeV":
        raise ValueError("Unsupported prepared data schema/units")
    if manifest["settings"] != preparation_settings(config):
        raise ValueError("Preparation settings differ: run prepare again or restore matching data/selection settings")
    csv_path = prepared/"features.csv"
    if sha256(csv_path) != manifest["features_sha256"]:
        raise ValueError("Prepared feature checksum mismatch")
    frame = pd.read_csv(csv_path)
    validate_frame(frame,config.training.features)
    output = new_output(output)
    result = train_models(frame,config)
    selected = result.predictions[result.predictions.score > config.training.threshold]
    window = config.statistics.mass_window_gev
    systematic = config.statistics.background_fractional_systematic
    summary = {"model":config.training.model,"weighted_oof_auc":result.oof_auc,
               "threshold":config.training.threshold,"folds":result.fold_metrics,
               "hyperparameter_tuning":result.tuning,
               "baseline":summarize_mc(frame,window,systematic,config.data.fraction),
               "after_ml":summarize_mc(selected,window,systematic,config.data.fraction),
               "warnings":["Starter workflow; not validated to reproduce original paper results.",
                           "Fixed features/hyperparameters/threshold must be chosen before evaluation.",
                           "Z_proxy is an approximate expected MC count measure, not an observed discovery significance.",
                           "Data scores average fold models; MC scores are out-of-fold. Their responses need validation before observed inference.",
                           "legacy_absolute weight mode reproduces source abs() convention; physics review required." if config.data.weight_mode == "legacy_absolute" else "Signed mode: training refuses negative weights."]}
    result.predictions.to_csv(output/"predictions.csv",index=False)
    write_json(output/"summary.json",summary)
    write_json(output/"config.json",config.as_dict())
    write_json(output/"provenance.json",{"environment":environment(),"prepared_manifest":manifest})
    save_mass_plot(frame,output/"mass_before.png","Before ML — starter workflow")
    save_mass_plot(selected,output/"mass_after.png","After ML — starter workflow")
    save_roc_plot(result.predictions,output/"roc.png")
    save_detailed_mass_plot(result.predictions, output/"real_data_m4l.png",
                            config.training.threshold, config.training.model)
    return output
