#!/usr/bin/env python3
"""Blind isolation scan using fixed no-isolation ATLAS-style quadruplets."""
import argparse
from dataclasses import replace
from pathlib import Path

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from higgs_lab.config import load_config
from higgs_lab.datasets import iter_batches, resolve_samples
from higgs_lab.selection import select_events

TRACK = (0.15, 0.20, 0.25)
E_CALO = (0.20, 0.25, 0.30)
MU_CALO = 0.30
SIDEBANDS = ((105.0, 115.0), (130.0, 160.0))
SIGNAL_WINDOW = (115.0, 130.0)

def in_regions(values, regions):
    mask=np.zeros(len(values),dtype=bool)
    for low,high in regions: mask|=(values>=low)&(values<high)
    return mask

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    config=load_config(args.config)
    # Select the physics quadruplet without isolation, then evaluate a fixed,
    # predefined isolation grid. The signal window is never used in the
    # data-side working-point choice.
    loose=replace(config,selection=replace(config.selection,
        track_isolation_max=1e9,electron_calo_isolation_max=1e9,
        muon_calo_isolation_max=1e9,retained_mass_range_gev=(80.,250.)))
    points=[(track,e_calo) for track in TRACK for e_calo in E_CALO]
    totals={(track,e):{"data_sb":0.,"bkg_sb":0.,"bkg_sb_w2":0.,
                       "signal_sr":0.,"data_sr":0.}
            for track,e in points}
    signal_noiso_sr=0.
    for sample,urls in resolve_samples(loose):
        print(f"Scanning {sample['name']}",flush=True)
        for url in urls:
            for events in iter_batches(url,loose,sample["role"]):
                selected,_=select_events(events,loose,sample["role"])
                if not len(selected): continue
                mass=ak.to_numpy(selected.mass)
                weight=ak.to_numpy(selected.totalWeight)
                pid=ak.to_numpy(selected.lep_type)
                pt=np.maximum(ak.to_numpy(selected.lep_pt),1e-9)
                track_ratio=ak.to_numpy(selected.lep_ptvarcone30)/pt
                calo_ratio=ak.to_numpy(selected.lep_topoetcone20)/pt
                sideband=in_regions(mass,SIDEBANDS)
                signal_region=in_regions(mass,(SIGNAL_WINDOW,))
                if sample["role"]=="signal": signal_noiso_sr+=weight[signal_region].sum()
                for track,e_calo in points:
                    iso=np.all((track_ratio<track)&
                        (calo_ratio<np.where(pid==13,MU_CALO,e_calo)),axis=1)
                    values=totals[(track,e_calo)]
                    if sample["role"]=="data":
                        values["data_sb"]+=np.count_nonzero(iso&sideband)
                        values["data_sr"]+=np.count_nonzero(iso&signal_region)
                    elif sample["role"]=="background":
                        chosen=weight[iso&sideband]
                        values["bkg_sb"]+=chosen.sum(); values["bkg_sb_w2"]+=np.square(chosen).sum()
                    else:
                        values["signal_sr"]+=weight[iso&signal_region].sum()
    rows=[]
    for (track,e_calo),values in totals.items():
        uncertainty=np.sqrt(values["data_sb"]+values["bkg_sb_w2"]+(0.30*values["bkg_sb"])**2)
        row={"track_iso_max":track,"electron_calo_iso_max":e_calo,
             "muon_calo_iso_max":MU_CALO,**values,
             "signal_efficiency_vs_no_iso":values["signal_sr"]/signal_noiso_sr,
             "sideband_data_mc_ratio":values["data_sb"]/values["bkg_sb"] if values["bkg_sb"]>0 else np.nan,
             "sideband_pull":(values["data_sb"]-values["bkg_sb"])/uncertainty if uncertainty>0 else np.nan}
        rows.append(row)
    frame=pd.DataFrame(rows)
    eligible=frame[frame.signal_efficiency_vs_no_iso>=.95]
    chosen=(eligible if len(eligible) else frame).iloc[
        (eligible if len(eligible) else frame).sideband_pull.abs().argmin()]
    frame["selected_by_rule"]=False
    frame.loc[(frame.track_iso_max==chosen.track_iso_max)&
              (frame.electron_calo_iso_max==chosen.electron_calo_iso_max),"selected_by_rule"]=True
    output=Path(args.output); output.mkdir(parents=True,exist_ok=False)
    frame.to_csv(output/"isolation_scan.csv",index=False)
    pd.DataFrame([{"selection_rule":"minimum absolute sideband pull among points with >=95% no-isolation Higgs-MC efficiency",
                   "sidebands":"105-115 and 130-160 GeV","signal_window_blinded_for_choice":"115-130 GeV",
                   "track_iso_max":chosen.track_iso_max,"electron_calo_iso_max":chosen.electron_calo_iso_max,
                   "muon_calo_iso_max":MU_CALO}]).to_csv(output/"chosen_working_point.csv",index=False)
    fig,axes=plt.subplots(1,3,figsize=(16,4.8))
    for ax,column,title in zip(axes,
        ["signal_efficiency_vs_no_iso","sideband_data_mc_ratio","sideband_pull"],
        ["Higgs MC efficiency","Sideband data/MC","Sideband pull"]):
        table=frame.pivot(index="electron_calo_iso_max",columns="track_iso_max",values=column)
        image=ax.imshow(table.values,origin="lower",aspect="auto",cmap="viridis")
        ax.set_xticks(range(len(table.columns)),[f"{v:.2f}" for v in table.columns])
        ax.set_yticks(range(len(table.index)),[f"{v:.2f}" for v in table.index])
        ax.set(xlabel="Track relative-isolation maximum",ylabel="Electron calo-isolation maximum",title=title)
        for i in range(table.shape[0]):
            for j in range(table.shape[1]): ax.text(j,i,f"{table.values[i,j]:.2f}",ha="center",va="center",color="white")
        fig.colorbar(image,ax=ax)
    fig.suptitle("Isolation working-point scan (data signal window excluded from selection rule)")
    fig.tight_layout(); fig.savefig(output/"isolation_scan.png",dpi=180); plt.close(fig)
    print(frame.to_string(index=False)); print("CHOSEN",chosen.to_dict())

if __name__=="__main__": main()
