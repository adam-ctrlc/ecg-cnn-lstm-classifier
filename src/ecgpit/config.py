from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
NAMES = {"NORM": "Normal ECG", "MI": "Myocardial infarction", "STTC": "ST/T change",
         "CD": "Conduction disturbance", "HYP": "Hypertrophy"}


def resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def load_config(path="configs/default.yaml") -> dict:
    return yaml.safe_load(resolve(path).read_text())
