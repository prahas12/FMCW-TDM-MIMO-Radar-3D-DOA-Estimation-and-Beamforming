import numpy as np
from radar_pipeline.dsp.windows import get_window

def compute_doppler_fft(range_profile, config):
    # range_profile: (channels, chirps, range_bins)
    n_fft = config['processing'].get('doppler_fft_size', range_profile.shape[1])
    win_name = config['processing'].get('doppler_window', 'hann')
    
    win = get_window(win_name, range_profile.shape[1])
    data_win = range_profile * win[np.newaxis, :, np.newaxis]
    
    # FFT over slow time (axis=1)
    rd_map_full = np.fft.fft(data_win, n=n_fft, axis=1)
    # Shift doppler to center 0 velocity
    rd_map = np.fft.fftshift(rd_map_full, axes=1)
    return rd_map
