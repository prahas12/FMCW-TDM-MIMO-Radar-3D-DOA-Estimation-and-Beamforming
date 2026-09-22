import numpy as np
from radar_pipeline.radar.waveform import FMCWWaveform
from scipy import constants

def test_waveform_physics():
    config = {
        'radar': {
            'carrier_frequency': 77e9,
            'bandwidth': 4e9,
            'chirp_duration': 40e-6,
            'idle_time': 10e-6,
            'adc_sample_rate': 10e6,
            'adc_samples': 256,
            'num_chirps': 128,
            'tx_count': 3,
            'rx_count': 4
        }
    }
    wf = FMCWWaveform(config)
    
    assert np.isclose(wf.lambda_c, constants.c / 77e9)
    assert np.isclose(wf.slope, 4e9 / 40e-6)
    assert np.isclose(wf.range_res, constants.c / (2 * 4e9))
