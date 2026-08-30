"""Portable provenance without pickled models or machine-specific paths."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from datetime import datetime, timezone

def sha256(path):
    h = hashlib.sha256()
    with open(path,"rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def environment():
    packages = {}
    for name in ["higgs-analysis-lab","numpy","pandas","matplotlib","scikit-learn","uproot","awkward","vector","atlasopenmagic"]:
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    return {"python": platform.python_version(),"platform": platform.platform(),"packages": packages,
            "created_utc": datetime.now(timezone.utc).isoformat()}

def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")

def preparation_settings(config):
    # Normalize tuples/lists before comparing with serialized JSON.
    return json.loads(json.dumps({"data":config.as_dict()["data"],"selection":config.as_dict()["selection"]}))

def new_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path
