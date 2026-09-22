import yaml
from dataclasses import dataclass
from typing import List

@dataclass
class TargetConfig:
    range: float
    velocity: float
    azimuth: float
    elevation: float
    rcs: float = 1.0

@dataclass
class RadarConfig:
    carrier_frequency: float
    bandwidth: float
    chirp_duration: float
    idle_time: float
    adc_sample_rate: float
    adc_samples: int
    num_chirps: int
    tx_count: int
    rx_count: int

@dataclass
class Config:
    radar: RadarConfig
    targets: List[TargetConfig]
    # other fields will be dictionaries for simplicity

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)
