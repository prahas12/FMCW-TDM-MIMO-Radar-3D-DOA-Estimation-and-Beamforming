import numpy as np
from radar_pipeline.radar.simulator import RadarSimulator
from radar_pipeline.dsp.range_fft import compute_range_fft
from radar_pipeline.dsp.doppler_fft import compute_doppler_fft
from radar_pipeline.detection.cfar import extract_peaks_cfar
from radar_pipeline.angle.music import compute_covariance, music_2d
from radar_pipeline.angle.bartlett import bartlett_2d
from radar_pipeline.angle.mvdr import mvdr_2d
from radar_pipeline.angle.steering import compute_steering_matrix
from radar_pipeline.angle.peak_detection import extract_peaks_2d
from radar_pipeline.localization.coordinates import spherical_to_cartesian


def _tdm_doppler_compensation(X, d_bin_centred, num_tx, doppler_fft_size, pri_chirp, lambda_c):
    """Compensate TDM-MIMO Doppler phase coupling across TX elements.

    In TDM-MIMO, each TX fires sequentially with a per-chirp interval of
    pri_chirp = chirp_duration + idle_time. A moving target accumulates an
    extra Doppler phase of 2*pi*fd*t_idx*pri_chirp across TX index t_idx.
    This function removes that phase using the estimated Doppler frequency
    derived from the CFAR-detected Doppler bin.
    """
    num_v = X.shape[0]
    num_rx = num_v // num_tx
    fd_est = d_bin_centred / (doppler_fft_size * num_tx * pri_chirp)

    X_comp = X.copy()
    for t_idx in range(num_tx):
        extra_phase = 2.0 * np.pi * fd_est * t_idx * pri_chirp
        correction = np.exp(-1j * extra_phase)
        for r_idx in range(num_rx):
            v_idx = t_idx * num_rx + r_idx
            X_comp[v_idx, :] *= correction

    return X_comp


def run_pipeline(config):
    sim = RadarSimulator(config)
    data = sim.generate_data_cube()

    r_fft = compute_range_fft(data, config)
    d_fft = compute_doppler_fft(r_fft, config)

    cfar_targets = extract_peaks_cfar(d_fft, config)

    final_targets = []
    v_pos = sim.array.virtual_pos
    lambda_c = sim.waveform.lambda_c
    num_tx = sim.array.num_tx
    n_chirps = sim.waveform.n_chirps
    t_chirp = sim.waveform.t_chirp_total
    doppler_fft_size = config['processing']['doppler_fft_size']
    range_fft_size = config['processing']['range_fft_size']

    az_min = config['angle_estimation']['azimuth_scan_min']
    az_max = config['angle_estimation']['azimuth_scan_max']
    az_res = config['angle_estimation']['azimuth_scan_res']
    el_min = config['angle_estimation']['elevation_scan_min']
    el_max = config['angle_estimation']['elevation_scan_max']
    el_res = config['angle_estimation']['elevation_scan_res']

    az_grid = np.arange(az_min, az_max + az_res, az_res)
    el_grid = np.arange(el_min, el_max + el_res, el_res)
    A_mat = compute_steering_matrix(az_grid, el_grid, v_pos, lambda_c)

    method = config['angle_estimation'].get('method', 'music')
    num_src_method = config['angle_estimation'].get('num_sources_method', 'mdl')

    for det in cfar_targets:
        r_bin = det['range_bin']
        d_bin = det['doppler_bin']

        # Doppler bin is stored from the zero-padded Doppler FFT (un-shifted).
        # Centre-shift it before using for velocity / phase compensation.
        d_bin_centred = d_bin - doppler_fft_size // 2

        # Extract multi-snapshot observation matrix X (M x N_chirps) at this range bin.
        X = r_fft[:, :, r_bin]   # shape (num_v, n_chirps)
        num_snapshots = X.shape[1]

        # Apply TDM-MIMO Doppler phase compensation so that the spatial covariance
        # reflects only the array geometry and target angles, not the inter-TX
        # Doppler phase accumulated during TDM transmission.
        X_comp = _tdm_doppler_compensation(
            X, d_bin_centred, num_tx, doppler_fft_size, t_chirp, lambda_c
        )

        R = compute_covariance(X_comp)

        if num_src_method == 'mdl':
            from radar_pipeline.angle.source_estimation import estimate_num_sources
            num_src = estimate_num_sources(R, method='mdl', num_snapshots=num_snapshots)
        else:
            num_src = config['angle_estimation'].get('known_num_sources', 1)

        if method == 'music':
            spectrum = music_2d(R, A_mat, num_sources=num_src)
        elif method == 'mvdr':
            spectrum = mvdr_2d(R, A_mat)
        elif method == 'bartlett':
            spectrum = bartlett_2d(R, A_mat)
        else:
            raise ValueError(f"Unknown method: {method}")

        angles = extract_peaks_2d(spectrum, az_grid, el_grid, num_src)

        range_m = (r_bin * sim.waveform.c * sim.waveform.fs
                   / (2.0 * sim.waveform.slope * range_fft_size))
        # Effective PRI = num_tx * (chirp_duration + idle_time)
        pri = num_tx * t_chirp
        vel_m = d_bin_centred * (lambda_c / (2.0 * pri * doppler_fft_size))

        for az, el, pwr in angles:
            x, y, z = spherical_to_cartesian(range_m, az, el)
            final_targets.append({
                'range': float(range_m),
                'velocity': float(vel_m),
                'azimuth': float(az),
                'elevation': float(el),
                'x': float(x),
                'y': float(y),
                'z': float(z),
                'snr': float(det['snr']),
                'spectrum_power': float(pwr)
            })

    return final_targets, sim
