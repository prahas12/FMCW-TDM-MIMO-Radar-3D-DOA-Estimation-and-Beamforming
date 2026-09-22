import numpy as np

def calibrate_channels(data_cube, reference_mismatch=None):
    """
    Inverts the channel mismatch if known (ideal calibration).
    """
    if reference_mismatch is None:
        return data_cube
        
    correction = 1.0 / reference_mismatch
    shape = [-1] + [1] * (data_cube.ndim - 1)
    return data_cube * correction.reshape(*shape)
