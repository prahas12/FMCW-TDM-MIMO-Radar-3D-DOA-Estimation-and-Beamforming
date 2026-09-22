import numpy as np
from radar_pipeline.radar.mimo_virtual_array import MIMOArray

def test_virtual_array():
    config = {
        'radar': {'tx_count': 3, 'rx_count': 4},
        'array': {
            'tx_spacing_x': 2.0,
            'tx_spacing_y': 0.5,
            'rx_spacing_x': 0.5,
            'rx_spacing_y': 0.0
        }
    }
    lambda_c = 0.004
    arr = MIMOArray(config, lambda_c)
    
    assert arr.num_tx == 3
    assert arr.num_rx == 4
    assert arr.virtual_pos.shape == (12, 3)
    
    # Check uniqueness
    unique_pos = np.unique(arr.virtual_pos, axis=0)
    assert len(unique_pos) == 12
