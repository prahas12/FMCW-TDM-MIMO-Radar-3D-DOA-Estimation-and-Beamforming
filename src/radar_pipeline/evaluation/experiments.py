import os
import json
import copy
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from radar_pipeline.pipeline import run_pipeline
from radar_pipeline.radar.simulator import RadarSimulator
from radar_pipeline.dsp.range_fft import compute_range_fft
from radar_pipeline.dsp.doppler_fft import compute_doppler_fft
from radar_pipeline.detection.cfar import extract_peaks_cfar
from radar_pipeline.angle.steering import compute_steering_matrix
from radar_pipeline.angle.bartlett import bartlett_2d
from radar_pipeline.angle.music import compute_covariance, music_2d
from radar_pipeline.config.loader import load_config
from radar_pipeline.calibration.channel_calibration import calibrate_channels


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def _make_config(base, overrides):
    """Deep-copy base config and apply overrides dict."""
    cfg = copy.deepcopy(base)
    for key, val in overrides.items():
        if isinstance(val, dict) and key in cfg:
            cfg[key].update(val)
        else:
            cfg[key] = val
    return cfg


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def run_baseline(config):
    ensure_dir("results/baseline")
    start = time.time()
    targets, sim = run_pipeline(config)
    runtime = time.time() - start

    save_json("results/baseline/baseline_config.json", config)
    save_json("results/baseline/ground_truth.json", config.get('targets', []))

    clean_targets = [{k: float(v) for k, v in t.items()} for t in targets]
    save_json("results/baseline/target_estimates.json", clean_targets)
    save_json("results/baseline/metrics.json", {
        "runtime_seconds": runtime,
        "detected_count": len(targets)
    })

    # Point cloud plot
    from radar_pipeline.visualization.point_cloud import plot_3d_targets
    plot_3d_targets(clean_targets, title="Baseline 3D Detections", save_path="results/baseline/3d_targets.png")

    return {"baseline_targets": len(targets), "runtime": runtime}


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def run_resolution(config):
    ensure_dir("results/resolution")
    # Sweep separation up to 35 deg because the 3dB beamwidth of 1.75 lambda aperture is ~29 deg.
    separations = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0]
    methods = ['bartlett', 'mvdr', 'music']
    trials = 10
    
    # Store success counts per method
    success_counts = {m: {sep: 0 for sep in separations} for m in methods}

    # Use a finer grid for better peak separation
    az_grid = np.arange(-50, 51, 0.5)
    el_grid = np.array([0.0])
    
    # Precompute steering matrix since array geometry is constant
    dummy_sim = RadarSimulator(config)
    A = compute_steering_matrix(az_grid, el_grid, dummy_sim.array.virtual_pos, dummy_sim.waveform.lambda_c)
    
    from scipy.ndimage import maximum_filter
    from radar_pipeline.angle.music import music_2d
    from radar_pipeline.angle.mvdr import mvdr_2d

    for sep in separations:
        for trial in range(trials):
            cfg = _make_config(config, {
                'simulation': {'snr_db': 25.0, 'seed': 300 + trial},
                'targets': [
                    {'range': 5.0, 'velocity': -0.5, 'azimuth': -sep / 2, 'elevation': 0.0, 'rcs': 1.0},
                    {'range': 5.0, 'velocity':  0.5, 'azimuth':  sep / 2, 'elevation': 0.0, 'rcs': 1.0},
                ],
            })
            sim = RadarSimulator(cfg)
            data = sim.generate_data_cube()
            r_fft = compute_range_fft(data, cfg)

            r_bin_expect = int(round(5.0 * 2 * sim.waveform.slope / sim.waveform.c / (sim.waveform.fs / cfg['processing']['range_fft_size'])))
            r_bin_expect = min(r_bin_expect, cfg['processing']['range_fft_size'] // 2 - 1)

            X = r_fft[:, :, r_bin_expect]
            R = compute_covariance(X, spatial_smoothing=False)
            
            true_az = sorted([-sep / 2, sep / 2])
            
            for method in methods:
                if method == 'music':
                    spec = music_2d(R, A, num_sources=2)
                elif method == 'mvdr':
                    spec = mvdr_2d(R, A)
                else:
                    spec = bartlett_2d(R, A)

                spec_1d = spec[0, :]
                lm = (maximum_filter(spec_1d, size=5) == spec_1d)
                peak_az = az_grid[lm & (spec_1d > spec_1d.max() - 15)]
                
                if len(peak_az) >= 2:
                    # check top 2 peaks
                    idx = np.argsort(spec_1d[lm & (spec_1d > spec_1d.max() - 15)])[-2:]
                    p_azs = sorted([peak_az[i] for i in idx])
                    # Relaxed tolerance for wide beamwidth array
                    if abs(p_azs[0] - true_az[0]) < 2.5 and abs(p_azs[1] - true_az[1]) < 2.5:
                        success_counts[method][sep] += 1

    success_rates = {m: [round(success_counts[m][sep] / trials, 3) for sep in separations] for m in methods}
    res = {'separations_deg': separations, 'trials': trials, 'success_rates': success_rates}
    save_json("results/resolution/summary.json", res)
    return res


# ---------------------------------------------------------------------------
# SNR Sweep
# ---------------------------------------------------------------------------

def run_snr(config):
    ensure_dir("results/snr")
    snrs = [-10, -5, 0, 5, 10, 20]
    methods = ['bartlett', 'music']
    trials = 20
    
    # Store results per method per snr
    sq_err_sum_counts = {m: {snr: 0.0 for snr in snrs} for m in methods}
    success_counts = {m: {snr: 0 for snr in snrs} for m in methods}

    # Use a finer grid (0.2 deg) to capture sub-degree RMSE
    az_grid = np.arange(-15, 35, 0.2)
    el_grid = np.array([0.0])

    # Precompute steering matrix
    dummy_sim = RadarSimulator(config)
    A = compute_steering_matrix(az_grid, el_grid, dummy_sim.array.virtual_pos, dummy_sim.waveform.lambda_c)

    # Success tolerance: peak must be within this many degrees of truth
    success_tol_deg = 3.0

    from radar_pipeline.angle.music import music_2d

    for snr in snrs:
        for trial in range(trials):
            # Sub-grid offset so true angle does not land on a grid point
            true_az = 10.33
            cfg = _make_config(config, {
                'simulation': {'snr_db': float(snr), 'seed': 400 + trial},
                'targets': [{'range': 5.0, 'velocity': 0.0, 'azimuth': true_az, 'elevation': 0.0, 'rcs': 1.0}],
            })
            sim = RadarSimulator(cfg)
            data = sim.generate_data_cube()
            r_fft = compute_range_fft(data, cfg)

            r_bin_expect = int(round(5.0 * 2 * sim.waveform.slope / sim.waveform.c / (sim.waveform.fs / cfg['processing']['range_fft_size'])))
            r_bin_expect = min(r_bin_expect, cfg['processing']['range_fft_size'] // 2 - 1)

            X = r_fft[:, :, r_bin_expect]
            R = compute_covariance(X, spatial_smoothing=False)
            
            for method in methods:
                if method == 'music':
                    spec = music_2d(R, A, num_sources=1)
                else:
                    spec = bartlett_2d(R, A)

                spec_1d = spec[0, :]
                peak_az = az_grid[np.argmax(spec_1d)]

                error = abs(peak_az - true_az)
                if error <= success_tol_deg:
                    success_counts[method][snr] += 1
                    sq_err_sum_counts[method][snr] += (peak_az - true_az) ** 2

    angle_success_rates = {m: [] for m in methods}
    rmse_results = {m: [] for m in methods}
    for m in methods:
        for snr in snrs:
            success = success_counts[m][snr]
            angle_success_rates[m].append(round(success / trials, 3))
            if success > 0:
                rmse_results[m].append(round(float(np.sqrt(sq_err_sum_counts[m][snr] / success)), 4))
            else:
                rmse_results[m].append(None)

    res = {'snrs_db': snrs, 'trials': trials, 'angle_success_rate': angle_success_rates, 'azimuth_rmse_deg': rmse_results}
    save_json("results/snr/summary.json", res)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for m in methods:
        axes[0].plot(snrs, [v if v is not None else 0 for v in rmse_results[m]], marker='o', label=m.upper())
        axes[1].plot(snrs, angle_success_rates[m], marker='s', label=m.upper())
    axes[0].set_xlabel("SNR (dB)"); axes[0].set_ylabel("Azimuth RMSE (deg)"); axes[0].legend()
    axes[1].set_xlabel("SNR (dB)"); axes[1].set_ylabel("Angle Success Rate"); axes[1].legend()
    axes[0].set_title("Azimuth RMSE vs SNR")
    axes[1].set_title("Angle Success Rate vs SNR")
    fig.tight_layout()
    fig.savefig("results/snr/snr_plot.png", dpi=100)
    plt.close(fig)
    return res


# ---------------------------------------------------------------------------
# Snapshot count sensitivity
# ---------------------------------------------------------------------------

def run_snapshots(config):
    ensure_dir("results/snapshots")
    snapshot_counts = [4, 8, 16, 32, 64, 128]
    trials = 20
    az_grid = np.arange(-40, 41, 0.5)
    el_grid = np.array([0.0])

    # Store results
    sq_err_music = {n: 0.0 for n in snapshot_counts}
    sq_err_bartlett = {n: 0.0 for n in snapshot_counts}
    valid_music = {n: 0 for n in snapshot_counts}
    valid_bartlett = {n: 0 for n in snapshot_counts}

    # Precompute steering matrix
    dummy_sim = RadarSimulator(config)
    A = compute_steering_matrix(az_grid, el_grid, dummy_sim.array.virtual_pos, dummy_sim.waveform.lambda_c)

    from radar_pipeline.angle.music import music_2d
    from scipy.ndimage import maximum_filter

    for trial in range(trials):
        cfg = _make_config(config, {
            'simulation': {'snr_db': 20.0, 'seed': 500 + trial},
            'targets': [
                {'range': 5.0, 'velocity': -0.5, 'azimuth': -6.0, 'elevation': 0.0, 'rcs': 1.0},
                {'range': 5.0, 'velocity':  0.5, 'azimuth':  6.0, 'elevation': 0.0, 'rcs': 1.0}
            ],
        })
        sim = RadarSimulator(cfg)
        data = sim.generate_data_cube()
        r_fft = compute_range_fft(data, cfg)

        r_bin_expect = int(round(5.0 * 2 * sim.waveform.slope / sim.waveform.c / (sim.waveform.fs / cfg['processing']['range_fft_size'])))
        r_bin_expect = min(r_bin_expect, cfg['processing']['range_fft_size'] // 2 - 1)

        for n_snap in snapshot_counts:
            n_use = min(n_snap, r_fft.shape[1])
            X = r_fft[:, :n_use, r_bin_expect]
            R = compute_covariance(X, spatial_smoothing=False)

            # MUSIC
            spec_music = music_2d(R, A, num_sources=2)[0, :]
            lm = (maximum_filter(spec_music, size=5) == spec_music)
            peak_az = az_grid[lm & (spec_music > spec_music.max() - 20)]
            if len(peak_az) >= 2:
                idx = np.argsort(spec_music[lm & (spec_music > spec_music.max() - 20)])[-2:]
                p_azs = sorted([peak_az[i] for i in idx])
                sq_err_music[n_snap] += (p_azs[0] - (-6.0)) ** 2 + (p_azs[1] - 6.0) ** 2
                valid_music[n_snap] += 2

            # Bartlett
            spec_bartlett = bartlett_2d(R, A)[0, :]
            lm_b = (maximum_filter(spec_bartlett, size=5) == spec_bartlett)
            peak_az_b = az_grid[lm_b & (spec_bartlett > spec_bartlett.max() - 20)]
            if len(peak_az_b) >= 2:
                idx = np.argsort(spec_bartlett[lm_b & (spec_bartlett > spec_bartlett.max() - 20)])[-2:]
                p_azs_b = sorted([peak_az_b[i] for i in idx])
                sq_err_bartlett[n_snap] += (p_azs_b[0] - (-6.0)) ** 2 + (p_azs_b[1] - 6.0) ** 2
                valid_bartlett[n_snap] += 2

    rmse_music = [round(float(np.sqrt(sq_err_music[n] / valid_music[n])) if valid_music[n] > 0 else 0.0, 4) for n in snapshot_counts]
    rmse_bartlett = [round(float(np.sqrt(sq_err_bartlett[n] / valid_bartlett[n])) if valid_bartlett[n] > 0 else 0.0, 4) for n in snapshot_counts]

    res = {
        'snapshot_counts': snapshot_counts,
        'trials': trials,
        'azimuth_rmse_music_deg': rmse_music,
        'azimuth_rmse_bartlett_deg': rmse_bartlett,
    }
    save_json("results/snapshots/summary.json", res)
    return res


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def run_calibration(config):
    """
    Simulate per-channel gain/phase mismatch, measure angle error before and
    after applying the known correction vector.
    """
    ensure_dir("results/calibration")

    az_grid = np.arange(-40, 41, 0.5)
    el_grid = np.array([0.0])
    true_az = 15.0
    trials = 20
    errors_uncal = []
    errors_cal = []

    rng = np.random.default_rng(99)

    for trial in range(trials):
        cfg = _make_config(config, {
            'simulation': {'snr_db': 25.0, 'seed': 600 + trial},
            'targets': [{'range': 5.0, 'velocity': 0.0, 'azimuth': true_az, 'elevation': 0.0, 'rcs': 1.0}],
        })
        sim = RadarSimulator(cfg)
        data = sim.generate_data_cube()
        r_fft = compute_range_fft(data, cfg)

        r_bin = int(round(
            5.0 * 2 * sim.waveform.slope / sim.waveform.c
            / (sim.waveform.fs / cfg['processing']['range_fft_size'])
        ))
        r_bin = min(r_bin, cfg['processing']['range_fft_size'] // 2 - 1)

        X_clean = r_fft[:, :, r_bin]

        # Apply random gain/phase mismatch
        gains = rng.normal(1.0, 0.15, X_clean.shape[0])
        phases = np.deg2rad(rng.normal(0.0, 15.0, X_clean.shape[0]))
        mismatch = gains * np.exp(1j * phases)
        X_impaired = X_clean * mismatch[:, np.newaxis]

        # Calibrated = apply inverse mismatch
        X_calibrated = calibrate_channels(X_impaired, mismatch)

        A = compute_steering_matrix(az_grid, el_grid, sim.array.virtual_pos, sim.waveform.lambda_c)

        for X, err_list in [(X_impaired, errors_uncal), (X_calibrated, errors_cal)]:
            R = compute_covariance(X)
            spec = bartlett_2d(R, A)
            peak_az = az_grid[np.argmax(spec[0, :])]
            err_list.append(float(abs(peak_az - true_az)))

    res = {
        'trials': trials,
        'true_azimuth_deg': true_az,
        'mean_error_uncalibrated_deg': round(float(np.mean(errors_uncal)), 4),
        'mean_error_calibrated_deg': round(float(np.mean(errors_cal)), 4),
        'std_error_uncalibrated_deg': round(float(np.std(errors_uncal)), 4),
        'std_error_calibrated_deg': round(float(np.std(errors_cal)), 4),
    }
    save_json("results/calibration/summary.json", res)
    return res


# ---------------------------------------------------------------------------
# Runtime Benchmark
# ---------------------------------------------------------------------------

def run_benchmark(config):
    ensure_dir("results/benchmark")
    methods = ['bartlett', 'mvdr', 'music']
    times = {m: [] for m in methods}
    repeats = 3

    for method in methods:
        cfg = _make_config(config, {
            'angle_estimation': {**config['angle_estimation'], 'method': method, 'num_sources_method': 'known', 'known_num_sources': 1},
        })
        for _ in range(repeats):
            t0 = time.perf_counter()
            run_pipeline(cfg)
            times[method].append(time.perf_counter() - t0)

    avg = {m: round(float(np.mean(v)), 4) for m, v in times.items()}
    std = {m: round(float(np.std(v)), 4) for m, v in times.items()}
    res = {'repeats': repeats, 'mean_runtime_seconds': avg, 'std_runtime_seconds': std}
    save_json("results/benchmark/summary.json", res)
    return res


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_all(config_path="configs/baseline.yaml"):
    config = load_config(config_path)
    ensure_dir("results")
    manifest = {}

    print("Running Baseline...")
    manifest['baseline'] = run_baseline(config)
    print("Running Resolution Experiment...")
    manifest['resolution'] = run_resolution(config)
    print("Running SNR Experiment...")
    manifest['snr'] = run_snr(config)
    print("Running Snapshot Experiment...")
    manifest['snapshots'] = run_snapshots(config)
    print("Running Calibration Experiment...")
    manifest['calibration'] = run_calibration(config)
    print("Running Benchmark...")
    manifest['benchmark'] = run_benchmark(config)

    save_json("results/manifest.json", manifest)
    print("All experiments completed. Results saved to results/manifest.json")


if __name__ == '__main__':
    run_all()
