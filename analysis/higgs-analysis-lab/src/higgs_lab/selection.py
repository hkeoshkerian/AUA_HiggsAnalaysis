"""Event cuts and weight calculation, independent of I/O and plotting."""
from itertools import combinations
import numpy as np
import awkward as ak
from .reconstruction import calc_mass

KINEMATIC_BRANCHES = [
    "lep_pt", "lep_eta", "lep_phi", "lep_e", "lep_charge", "lep_type",
    "trigE", "trigM", "lep_isTrigMatched", "lep_isLooseID",
    "lep_isMediumID", "lep_isLooseIso", "lep_n", "jet_n", "met", "met_phi"
]
ATLAS_BRANCHES = [
    "trigDE", "trigDM", "trigT", "trigDT", "lep_z0", "lep_d0",
    "lep_ptvarcone30", "lep_topoetcone20",
]
WEIGHT_BRANCHES = ["filteff", "kfac", "xsec", "mcWeight", "ScaleFactor_PILEUP",
                   "ScaleFactor_ELE", "ScaleFactor_MUON", "ScaleFactor_LepTRIGGER"]

def required_branches(config):
    return KINEMATIC_BRANCHES + (ATLAS_BRANCHES if config.selection.mode == "atlas_2017_fiducial" else [])

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

def _reference_selection(events, config, role):
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

def _pair_mass(pt, eta, phi, energy, i, j):
    px = pt * np.cos(phi); py = pt * np.sin(phi); pz = pt * np.sinh(eta)
    e = energy[i] + energy[j]
    x = px[i] + px[j]; y = py[i] + py[j]; z = pz[i] + pz[j]
    return float(np.sqrt(max(e*e - x*x - y*y - z*z, 0.0)))

def _dr(eta, phi, i, j):
    dphi = abs(phi[i] - phi[j])
    dphi = min(dphi, 2*np.pi-dphi)
    return float(np.hypot(eta[i]-eta[j], dphi))

def _atlas_quadruplets(events):
    """Choose one ATLAS-style SFOS quadruplet per event.

    Returns event indices, ordered lepton indices (Z1 then Z2), and cumulative
    event counts for the cuts that require a complete quadruplet.
    """
    mz = 91.1876
    pairings = ((0,1,2,3), (0,2,1,3), (0,3,1,2))
    stages = {name: 0 for name in (
        "two_sfos_pairs", "leading_lepton_pt", "z_pair_masses",
        "lepton_separation", "jpsi_veto")}
    chosen_events, chosen_leptons = [], []
    fields = ["lep_pt","lep_eta","lep_phi","lep_e","lep_charge","lep_type"]
    arrays = {name: ak.to_list(events[name]) for name in fields}
    for event_index in range(len(events)):
        pt=np.asarray(arrays["lep_pt"][event_index],float)
        eta=np.asarray(arrays["lep_eta"][event_index],float)
        phi=np.asarray(arrays["lep_phi"][event_index],float)
        energy=np.asarray(arrays["lep_e"][event_index],float)
        charge=np.asarray(arrays["lep_charge"][event_index],int)
        pid=np.asarray(arrays["lep_type"][event_index],int)
        candidates=[]
        passed={name:False for name in stages}
        for quad in combinations(range(len(pt)),4):
            for a,b,c,d in pairings:
                i,j,k,l=(quad[a],quad[b],quad[c],quad[d])
                if not (pid[i]==pid[j] and charge[i]+charge[j]==0 and
                        pid[k]==pid[l] and charge[k]+charge[l]==0):
                    continue
                passed["two_sfos_pairs"]=True
                order=sorted((i,j,k,l),key=lambda x:pt[x],reverse=True)
                if not (pt[order[0]]>20 and pt[order[1]]>15 and pt[order[2]]>10):
                    continue
                passed["leading_lepton_pt"]=True
                m1=_pair_mass(pt,eta,phi,energy,i,j); m2=_pair_mass(pt,eta,phi,energy,k,l)
                if abs(m2-mz)<abs(m1-mz):
                    i,j,k,l,m1,m2=k,l,i,j,m2,m1
                if not (50<m1<106 and 12<m2<115):
                    continue
                passed["z_pair_masses"]=True
                indices=(i,j,k,l)
                if any(_dr(eta,phi,x,y) <= (0.1 if pid[x]==pid[y] else 0.2)
                       for x,y in combinations(indices,2)):
                    continue
                passed["lepton_separation"]=True
                sfos=[_pair_mass(pt,eta,phi,energy,x,y)
                      for x,y in combinations(indices,2)
                      if pid[x]==pid[y] and charge[x]+charge[y]==0]
                if any(m<=5 for m in sfos):
                    continue
                passed["jpsi_veto"]=True
                z1=sorted((i,j),key=lambda x:pt[x],reverse=True)
                z2=sorted((k,l),key=lambda x:pt[x],reverse=True)
                types=[pid[x] for x in z1+z2]
                if all(x==13 for x in types): priority=0
                elif types[0]==types[1]==11: priority=1
                elif types[0]==types[1]==13: priority=2
                else: priority=3
                candidates.append(((priority,abs(m1-mz),abs(m2-mz)),z1+z2))
        for name,value in passed.items():
            stages[name]+=int(value)
        if candidates:
            candidates.sort(key=lambda item:item[0])
            chosen_events.append(event_index); chosen_leptons.append(candidates[0][1])
    return chosen_events, chosen_leptons, stages

def _atlas_selection(events, config, role):
    required = required_branches(config) + ([] if role == "data" else WEIGHT_BRANCHES + ["sum_of_weights"])
    missing=set(required)-set(events.fields)
    if missing:
        raise ValueError(f"Missing ROOT branches: {sorted(missing)}")
    rows=[]
    current=events
    def cut(mask,stage):
        nonlocal current
        current=current[mask]; rows.append({"stage":stage,"events":len(current)})
    rows.append({"stage":"input","events":len(current)})
    cut((current.lep_n>=4)&(ak.num(current.lep_pt,axis=1)>=4),"at_least_four_leptons")
    trigger=current.trigE|current.trigM|current.trigDE|current.trigDM|current.trigT|current.trigDT
    cut(trigger,"single_di_trilepton_trigger")
    cut(ak.sum(current.lep_isTrigMatched,axis=1)>=1,"trigger_match")
    pt=current.lep_pt; eta=np.abs(current.lep_eta); pid=current.lep_type
    acceptance=((pid==13)&(pt>5)&(eta<2.7))|((pid==11)&(pt>7)&(eta<2.47))
    identification=((pid==11)&current.lep_isLooseID)|((pid==13)&current.lep_isLooseID)
    # The compact Open Data tuple stores an absolute longitudinal coordinate,
    # not the primary-vertex-relative z0 used by the paper. Applying the
    # published 0.5 mm requirement to this field would reject valid data.
    # The available cosmic-muon transverse-impact requirement is retained.
    impact=(pid!=13)|(np.abs(current.lep_d0)<1.0)
    track_iso=(current.lep_ptvarcone30/np.maximum(pt,1e-9) <
               config.selection.track_isolation_max)
    calo_limit=ak.where(pid==13,config.selection.muon_calo_isolation_max,
                        config.selection.electron_calo_isolation_max)
    calo_iso=current.lep_topoetcone20/np.maximum(pt,1e-9)<calo_limit
    good=acceptance&identification&impact&track_iso&calo_iso
    event_good=ak.sum(good,axis=1)>=4
    good=good[event_good]
    cut(event_good,"lepton_acceptance_id_isolation_impact")
    lepton_fields=[name for name in required_branches(config) if name.startswith("lep_") and name!="lep_n"]
    masked_leptons={name:current[name][good] for name in lepton_fields}
    for name in lepton_fields:
        current=ak.with_field(current,masked_leptons[name],name)
    current=ak.with_field(current,ak.sum(good,axis=1),"lep_n")
    event_indices, lepton_indices, stages=_atlas_quadruplets(current)
    current=current[np.asarray(event_indices,dtype=int)]
    index_array=ak.Array(lepton_indices)
    for name in lepton_fields:
        current=ak.with_field(current,current[name][index_array],name)
    current=ak.with_field(current,np.full(len(current),4),"lep_n")
    rows.extend({"stage":name,"events":count} for name,count in stages.items())
    current=ak.with_field(current,calc_mass(current.lep_pt,current.lep_eta,current.lep_phi,current.lep_e),"mass")
    mass_low,mass_high=config.selection.retained_mass_range_gev
    cut((current.mass>mass_low)&(current.mass<mass_high),
        f"m4l_{mass_low:g}_{mass_high:g}_GeV")
    weights=np.ones(len(current)) if role=="data" else calculate_weights(current,config.data.luminosity_fb,config.data.weight_mode)
    current=ak.with_field(current,weights,"totalWeight")
    return current,rows

def select_events(events, config, role):
    """Apply the configured reference or ATLAS-2017-style selection."""
    if role not in {"data","signal","background"}:
        raise ValueError("Invalid sample role")
    if config.selection.mode == "atlas_2017_fiducial":
        return _atlas_selection(events,config,role)
    return _reference_selection(events,config,role)
