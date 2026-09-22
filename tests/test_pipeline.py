import numpy as np
from radar_pipeline.config.loader import load_config
from radar_pipeline.pipeline import run_pipeline
import yaml

def test_baseline_pipeline(tmp_path):
    config = {
        'radar': {
            'carrier_frequency': 77e9, 'bandwidth': 4e9, 'chirp_duration': 40e-6,
            'idle_time': 10e-6, 'adc_sample_rate': 10e6, 'adc_samples': 256,
            'num_chirps': 32, 'tx_count': 3, 'rx_count': 4
        },
        'array': {
            'tx_spacing_x': 2, 'tx_spacing_y': 0.5, 'rx_spacing_x': 0.5, 'rx_spacing_y': 0
        },
        'targets': [
            {'range': 5.0, 'velocity': 0.0, 'azimuth': 0.0, 'elevation': 0.0, 'rcs': 1.0}
        ],
        'simulation': {'snr_db': 50.0, 'seed': 42},
        'processing': {'range_fft_size': 256, 'doppler_fft_size': 32, 'range_window': 'hann', 'doppler_window': 'hann'},
        'detection': {'cfar_type': 'ca', 'pfa': 1e-2, 'train_cells': 8, 'guard_cells': 2},
        'angle_estimation': {
            'azimuth_scan_min': -20, 'azimuth_scan_max': 20, 'azimuth_scan_res': 2.0,
            'elevation_scan_min': -10, 'elevation_scan_max': 10, 'elevation_scan_res': 2.0,
            'method': 'bartlett', 'known_num_sources': 1
        }
    }
    
    targets, sim = run_pipeline(config)
    
    assert len(targets) >= 1
    
    # Filter targets near ground truth range
    close_targets = [t for t in targets if abs(t['range'] - 5.0) < 0.5]
    assert len(close_targets) > 0
    
    # The actual peak should have the highest power
    best_t = max(close_targets, key=lambda t: t['spectrum_power'])
    
    assert abs(best_t['range'] - 5.0) < 0.5
    assert abs(best_t['velocity'] - 0.0) < 1.0
    assert abs(best_t['azimuth'] - 0.0) < 5.0
