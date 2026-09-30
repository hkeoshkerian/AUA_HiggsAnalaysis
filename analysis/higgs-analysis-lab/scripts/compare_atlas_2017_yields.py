#!/usr/bin/env python3
"""Compare prepared 115--130 GeV yields with ATLAS JHEP10 (2017) 132."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PAPER = pd.DataFrame({
    "channel": ["4mu", "4e", "2e2mu", "2mu2e"],
    "paper_signal": [20.1, 10.6, 14.2, 10.8],
    "paper_zz": [9.8, 4.4, 7.1, 4.6],
    "paper_other": [1.3, 1.3, 1.0, 1.4],
    "paper_expected": [31.2, 16.3, 22.3, 16.8],
    "paper_observed": [33, 16, 32, 21],
})

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--prepared",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    source=Path(args.prepared)/"preselection_mass.csv"
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    frame=pd.read_csv(source)
    frame=frame[(frame.mass>115.0)&(frame.mass<130.0)].copy()
    channels=PAPER.channel.tolist()
    rows=[]
    for channel in channels:
        selected=frame[frame.channel==channel]
        def y(mask): return float(selected.loc[mask,"weight"].sum())
        signal=y(selected.role=="signal")
        zz=y(selected["sample"].str.contains("ZZ",regex=False))
        other=y((selected.role=="background")&~selected["sample"].str.contains("ZZ",regex=False))
        observed=int((selected.role=="data").sum())
        rows.append({"channel":channel,"open_data_signal":signal,"open_data_zz":zz,
                     "open_data_other":other,"open_data_expected":signal+zz+other,
                     "open_data_observed":observed})
    result=PAPER.merge(pd.DataFrame(rows),on="channel")
    totals={"channel":"Total"}
    for column in result.columns[1:]: totals[column]=result[column].sum()
    result=pd.concat([result,pd.DataFrame([totals])],ignore_index=True)
    for component in ("signal","zz","other","expected","observed"):
        result[f"{component}_ratio_to_paper"]=result[f"open_data_{component}"]/result[f"paper_{component}"]
    result.to_csv(output/"atlas_yield_comparison.csv",index=False)
    shown=["channel","paper_signal","open_data_signal","paper_zz","open_data_zz",
           "paper_other","open_data_other","paper_expected","open_data_expected",
           "paper_observed","open_data_observed"]
    header="| "+" | ".join(shown)+" |\n"
    separator="| "+" | ".join(["---"]*len(shown))+" |\n"
    body="".join("| "+" | ".join(str(value) if isinstance(value,str) else f"{value:.2f}"
                 for value in row)+" |\n" for row in result[shown].itertuples(index=False,name=None))
    (output/"atlas_yield_comparison.md").write_text(header+separator+body)

    labels=result.channel.tolist(); x=np.arange(len(labels)); width=.36
    fig,axes=plt.subplots(1,2,figsize=(14,5.5))
    for ax,kind,title in [(axes[0],"expected","Total expected yield"),(axes[1],"observed","Observed data")]:
        ax.bar(x-width/2,result[f"paper_{kind}"],width,label="ATLAS paper",color="#555555")
        ax.bar(x+width/2,result[f"open_data_{kind}"],width,label="Open Data reproduction",color="#168aad")
        ax.set_xticks(x,labels); ax.set_ylabel("Events in 115–130 GeV"); ax.set_title(title)
        ax.grid(axis="y",alpha=.25); ax.legend()
    fig.suptitle("ATLAS 36.1 fb$^{-1}$ yield comparison")
    fig.tight_layout(); fig.savefig(output/"atlas_yield_comparison.png",dpi=180); plt.close(fig)

if __name__=="__main__": main()
