import numpy as np
from scipy.ndimage import maximum_filter

def extract_peaks_2d(spectrum, az_grid, el_grid, num_peaks):
    # NMS to avoid duplicate mirrored peaks
    local_max = (maximum_filter(spectrum, size=5) == spectrum)
    peaks = np.where(local_max, spectrum, -np.inf)
    
    flat_indices = np.argsort(peaks.flatten())[::-1]
    
    results = []
    for idx in flat_indices[:num_peaks]:
        el_idx, az_idx = np.unravel_index(idx, spectrum.shape)
        if local_max[el_idx, az_idx]:
            results.append((az_grid[az_idx], el_grid[el_idx], spectrum[el_idx, az_idx]))
            
    return results
