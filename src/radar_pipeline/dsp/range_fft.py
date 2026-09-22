import numpy as np
from radar_pipeline.dsp.windows import get_window

def compute_range_fft(data, config):
    # data: (channels, chirps, samples)
    n_fft = config['processing'].get('range_fft_size', data.shape[2])
    win_name = config['processing'].get('range_window', 'hann')
    
    win = get_window(win_name, data.shape[2])
    data_win = data * win[np.newaxis, np.newaxis, :]
    
    range_profile = np.fft.fft(data_win, n=n_fft, axis=2)
    return range_profile[:, :, :n_fft//2]
