"""Validated configuration; no third-party dependencies needed to load it."""
from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
import tomllib

models = {
    "xgboost",
    "lightgbm",
    "random_forest",
    "mlp",
    "logistic_regression",
    "gaussian_nb",
    "qda",
}

@dataclass(frozen=True)
class DataConfig:
    release: str = "2025e-13tev-beta"
    skim: str = "exactly4lep"
    luminosity_fb: float = 36.6
    fraction: float = 1.0
    step_size: str = "100 MB"
    weight_mode: str = "legacy_absolute"

@dataclass(frozen=True)
class SelectionConfig:
    pt_min_gev: tuple = (20.0, 15.0, 10.0)
    # False reproduces the reference script, which indexes stored positions.
    sort_leptons_by_pt: bool = False

@dataclass(frozen=True)
class TrainingConfig:
    model: str = "logistic_regression"
    folds: int = 5
    seed: int = 42
    threshold: float = 0.65
    features: tuple = ("mz1", "mz2", "ptz1", "ptz2", "pt4l", "met", "jet_n")

@dataclass(frozen=True)
class StatisticsConfig:
    mass_window_gev: tuple = (110.0, 135.0)
    background_fractional_systematic: float = 0.30

@dataclass(frozen=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)

    def as_dict(self):
        return asdict(self)

    def validate(self):
        d, s, t, st = self.data, self.selection, self.training, self.statistics
        if not (math.isfinite(d.luminosity_fb) and d.luminosity_fb > 0):
            raise ValueError("luminosity_fb must be positive and finite")
        if not 0 < d.fraction <= 1:
            raise ValueError("fraction must be in (0, 1]")
        if d.skim != "exactly4lep":
            raise ValueError("This starter supports only the exactly4lep skim")
        if d.weight_mode not in {"legacy_absolute", "signed"}:
            raise ValueError("weight_mode must be legacy_absolute or signed")
        if len(s.pt_min_gev) not in {3, 4} or any(not math.isfinite(v) or v < 0 for v in s.pt_min_gev):
            raise ValueError("Provide three or four finite nonnegative pT thresholds")
        if type(s.sort_leptons_by_pt) is not bool:
            raise ValueError("sort_leptons_by_pt must be true or false")
        if t.model not in models:
            raise ValueError(f"Supported models: {', '.join(sorted(models))}")
        if type(t.folds) is not int or t.folds < 2 or type(t.seed) is not int or not 0 <= t.seed < 2**32:
            raise ValueError("folds must be an integer >= 2 and seed an unsigned 32-bit integer")
        if not 0 < t.threshold < 1:
            raise ValueError("threshold must be in (0, 1)")
        if not t.features or len(set(t.features)) != len(t.features):
            raise ValueError("features must be nonempty and unique")
        if len(st.mass_window_gev) != 2 or not all(math.isfinite(v) for v in st.mass_window_gev) or not st.mass_window_gev[0] < st.mass_window_gev[1]:
            raise ValueError("mass_window_gev must contain finite increasing bounds")
        if not math.isfinite(st.background_fractional_systematic) or st.background_fractional_systematic < 0:
            raise ValueError("background systematic must be finite and nonnegative")
        return self

def load_config(path):
    raw = tomllib.loads(Path(path).read_text())
    classes = {"data": DataConfig, "selection": SelectionConfig,
               "training": TrainingConfig, "statistics": StatisticsConfig}
    unknown = set(raw) - classes.keys()
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    try:
        config = Config(**{k: classes[k](**v) for k, v in raw.items()})
    except TypeError as exc:
        raise ValueError(f"Invalid configuration key: {exc}") from exc
    return config.validate()
