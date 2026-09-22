import numpy as np
from scipy.ndimage import uniform_filter, maximum_filter

def ca_cfar_2d(rd_map_power, train_c, guard_c, pfa):
    num_dop, num_range = rd_map_power.shape
    N = (2*train_c + 2*guard_c + 1)**2 - (2*guard_c + 1)**2
    alpha = N * (pfa**(-1.0/N) - 1)
    
    window_size = 2*train_c + 2*guard_c + 1
    guard_size = 2*guard_c + 1
    
    P_total = uniform_filter(rd_map_power, size=window_size, mode='constant', cval=0.0) * (window_size**2)
    P_guard = uniform_filter(rd_map_power, size=guard_size, mode='constant', cval=0.0) * (guard_size**2)
    P_noise = (P_total - P_guard) / N
    
    threshold = alpha * P_noise
    detections = rd_map_power > threshold
    return detections, P_noise

def extract_peaks_cfar(rd_map, config):
    rd_map_power = np.sum(np.abs(rd_map)**2, axis=0)
    train_c = config['detection'].get('train_cells', 16)
    guard_c = config['detection'].get('guard_cells', 4)
    pfa = float(config['detection'].get('pfa', 1e-4))
    
    detections, P_noise = ca_cfar_2d(rd_map_power, train_c, guard_c, pfa)
    
    # NMS: only keep strictly local maxima within a 3x3 window
    local_max = (maximum_filter(rd_map_power, size=5) == rd_map_power)
    peaks = detections & local_max
    
    dop_indices, range_indices = np.where(peaks)
    
    targets = []
    for d, r in zip(dop_indices, range_indices):
        targets.append({
            'range_bin': int(r),
            'doppler_bin': int(d),
            'power': float(rd_map_power[d, r]),
            'snr': float(rd_map_power[d, r] / (P_noise[d, r] + 1e-12))
        })
    
    # Sort by SNR and limit to top 20 to prevent clutter overload
    targets.sort(key=lambda x: x['snr'], reverse=True)
    return targets[:20]
