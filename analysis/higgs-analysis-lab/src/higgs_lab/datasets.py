"""Network I/O is explicit; no download or environment changes at import time."""
from .samples import SAMPLES

def resolve_samples(config):
    import atlasopenmagic as atom
    atom.set_release(config.data.release)
    definitions = {s["name"]: {k:v for k,v in s.items() if k not in {"name", "role"}} for s in SAMPLES}
    resolved = atom.build_dataset(definitions, skim=config.data.skim, protocol="https", cache=True)
    return [(sample, resolved[sample["name"]]["list"]) for sample in SAMPLES]

def iter_batches(url, config, role):
    import uproot
    from .selection import KINEMATIC_BRANCHES, WEIGHT_BRANCHES
    branches = KINEMATIC_BRANCHES + ([] if role == "data" else WEIGHT_BRANCHES + ["sum_of_weights"])
    with uproot.open(url) as root_file:
        tree = root_file["analysis"]
        missing = set(branches) - set(tree.keys())
        if missing:
            raise ValueError(f"Missing branches in {url}: {sorted(missing)}")
        stop = int(tree.num_entries * config.data.fraction)
        yield from tree.iterate(branches, library="ak", entry_stop=stop, step_size=config.data.step_size)
