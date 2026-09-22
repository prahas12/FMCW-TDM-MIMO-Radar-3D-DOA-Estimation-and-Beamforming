# FMCW TDM-MIMO Radar: 3D DOA Estimation and Beamforming

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![DSP](https://img.shields.io/badge/DSP-4B4B4B?style=for-the-badge)
![Radar](https://img.shields.io/badge/Radar-4B4B4B?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)


## 1. Project Overview

This project implements an end-to-end FMCW radar signal processing pipeline for 3D target localization using a Time-Division Multiplexed (TDM) MIMO array. The pipeline simulates baseband beat signals for configurable multi-target scenes, processes them through Range-Doppler FFT, CA-CFAR detection, TDM Doppler phase compensation, and 2D angle-of-arrival estimation to produce 3D Cartesian target coordinates.

Three spatial spectrum estimators are implemented and compared: Bartlett (conventional beamformer), MVDR (Capon), and MUSIC (subspace). Source enumeration uses the Minimum Description Length (MDL) criterion. The evaluation suite measures angular resolution, noise sensitivity, snapshot convergence, calibration robustness, and execution runtime across Monte Carlo trials.

## 2. Key Features

- TDM-MIMO beat signal simulation with per-chirp, per-transmitter slow-time phase progression.
- Virtual array synthesis from physical TX/RX antenna positions.
- TDM Doppler phase compensation to remove velocity-induced inter-TX phase offsets before angle estimation.
- 2D CA-CFAR detection with non-maximum suppression on the Range-Doppler map.
- Multi-snapshot temporal covariance estimation from the chirp dimension.
- MDL-based automatic source enumeration.
- Bartlett, MVDR, and MUSIC 2D angle estimators.
- Monte Carlo evaluation suite for resolution, SNR, snapshot count, calibration, and runtime benchmarking.

## 3. Processing Pipeline

1. Parse target scene from YAML configuration.
2. Simulate FMCW beat signals with TDM-MIMO transmission timing.
3. Compute Range FFT (windowed) across fast-time samples.
4. Compute Doppler FFT (windowed) across slow-time chirps.
5. Apply 2D CA-CFAR detection and non-maximum suppression.
6. For each detected Range-Doppler bin, extract the multi-snapshot spatial observation matrix from the Range FFT output.
7. Compensate TDM-MIMO Doppler phase coupling across TX elements.
8. Estimate the sample spatial covariance matrix.
9. Estimate number of sources via MDL.
10. Compute the 2D spatial spectrum (Bartlett, MVDR, or MUSIC) over an azimuth-elevation grid.
11. Extract angle peaks and convert (Range, Azimuth, Elevation) to Cartesian (x, y, z).

## 4. System Configuration

The baseline configuration (`configs/baseline.yaml`) defines:

| Parameter | Value |
| :-- | :-- |
| Carrier frequency | 77.0 GHz |
| Bandwidth | 4.0 GHz |
| Chirp duration | 40.0 us |
| Idle time | 10.0 us |
| ADC sample rate | 10.0 MHz |
| ADC samples per chirp | 256 |
| Number of chirps | 128 |
| TX antennas | 3 |
| RX antennas | 4 |
| Virtual channels | 12 |
| TX spacing (x) | 2 half-wavelengths |
| TX spacing (y) | 0.5 half-wavelengths |
| RX spacing (x) | 0.5 half-wavelengths |
| Range FFT size | 512 |
| Doppler FFT size | 256 |
| CFAR false alarm rate | 1e-4 |
| Angle estimation method | MUSIC |
| Source enumeration | MDL |

The baseline scene contains two targets:

| Target | Range | Velocity | Azimuth | Elevation | RCS |
| :-- | :-- | :-- | :-- | :-- | :-- |
| 1 | 5.0 m | 2.0 m/s | 10.0 deg | 5.0 deg | 1.0 |
| 2 | 6.0 m | -1.0 m/s | -15.0 deg | 10.0 deg | 2.0 |

## 5. Algorithms

**Bartlett**: Computes the conventional beamformer output as the quadratic form of the sample covariance matrix with the steering vector at each look direction.

**MVDR (Capon)**: Minimizes the total array output power subject to a unity-gain constraint in the look direction, providing improved interference rejection relative to Bartlett.

**MUSIC**: Decomposes the covariance matrix via eigendecomposition, separates signal and noise subspaces using MDL, and evaluates the inverse projection of each steering vector onto the noise subspace.

## 6. Repository Structure

```text
.
├── configs/
│   └── baseline.yaml          # Radar and scene configuration
├── src/
│   └── radar_pipeline/
│       ├── cli.py             # Command-line interface
│       ├── pipeline.py        # Main processing pipeline
│       ├── radar/             # Waveform, simulator, MIMO array, scene
│       ├── dsp/               # Range FFT, Doppler FFT, windowing
│       ├── detection/         # CA-CFAR detector
│       ├── angle/             # Bartlett, MVDR, MUSIC, steering vectors
│       ├── calibration/       # Channel gain/phase calibration
│       ├── evaluation/        # Monte Carlo experiment suite
│       ├── localization/      # Spherical to Cartesian conversion
│       └── visualization/     # 3D point cloud plotting
├── tests/                     # Unit tests
└── results/                   # Generated experiment outputs
```

## 7. Requirements

- Python 3.9 or later
- NumPy, SciPy, Matplotlib, PyYAML (installed automatically)
- pytest (for running tests, installed with `.[dev]`)

## 8. Installation

```bash
git clone https://github.com/prahas12/FMCW-TDM-MIMO-Radar-3D-DOA-Estimation-and-Beamforming.git
cd FMCW-TDM-MIMO-Radar-3D-DOA-Estimation-and-Beamforming
python -m venv .venv
```

Activate the virtual environment:

- Windows: `.venv\Scripts\activate`
- Linux/macOS: `source .venv/bin/activate`

Install the package:

```bash
pip install -e .[dev]
```

## 9. Quick Start

```bash
radar_pipeline run --config configs/baseline.yaml
```

This runs the full pipeline on the baseline 2-target scene and prints detected target parameters to the console.

## 10. Baseline Execution

```bash
radar_pipeline run --config configs/baseline.yaml
```

Expected output:

```
Detected 2 targets.
Target 1: R=6.00m, V=-1.01m/s, Az=-15.0, El=10.0, SNR=158843.8
Target 2: R=5.01m, V=1.98m/s, Az=10.0, El=5.0, SNR=136538.7
Saved 3D point cloud visualization to results/baseline/3d_targets.png
```

The estimated angles match the configured ground truth (Az=-15, El=10 and Az=10, El=5). Range and velocity estimates are within one FFT bin of the true values.

## 11. Running Experiments

```bash
radar_pipeline experiment --config configs/baseline.yaml
```

This executes the full Monte Carlo evaluation suite: baseline validation, angular resolution sweep, SNR sweep, snapshot count sweep, calibration test, and runtime benchmark. All results are written to `results/`. The suite completes in approximately 15 to 25 seconds on a typical consumer CPU.

## 12. Output Files and Results

After running the experiment suite, the following files are generated:

| File | Contents |
| :-- | :-- |
| `results/manifest.json` | Combined quantitative summary of all experiments |
| `results/baseline/target_estimates.json` | Estimated target parameters from the baseline run |
| `results/baseline/ground_truth.json` | Configured ground truth target parameters |
| `results/baseline/3d_targets.png` | 3D scatter plot of detected targets |
| `results/resolution/summary.json` | Resolution experiment success rates |
| `results/resolution/resolution_plot.png` | Resolution success rate plot |
| `results/snr/summary.json` | SNR experiment RMSE and angle success rates |
| `results/snr/snr_plot.png` | RMSE vs SNR plot |
| `results/snapshots/summary.json` | Snapshot count experiment RMSE |
| `results/snapshots/snapshot_plot.png` | Snapshot convergence plot |
| `results/calibration/summary.json` | Calibration experiment error statistics |
| `results/benchmark/summary.json` | Per-algorithm runtime statistics |

## 13. Quantitative Results

All values below are taken directly from the generated `results/manifest.json`.

### Baseline Detection

The pipeline detects both configured targets. Estimated angles match the ground truth exactly at the 1-degree scan grid resolution.

### Angular Resolution of Spatial Estimators (10 trials per separation, SNR = 25 dB)

This evaluates the angular resolution of Bartlett, MVDR, and MUSIC independently of the full detection pipeline. Two equal-RCS targets are placed at the same range with different velocities (-0.5, +0.5 m/s), separated in azimuth. Success rate is the fraction of trials where the estimator correctly resolves both peaks within the known range bin.

| Separation | Bartlett | MVDR | MUSIC |
| :-- | :-- | :-- | :-- |
| 5 deg | 0.0 | 0.0 | 0.0 |
| 10 deg | 0.0 | 0.0 | 0.0 |
| 15 deg | 0.0 | 1.0 | 1.0 |
| 20 deg | 0.0 | 1.0 | 1.0 |
| 25 deg | 0.0 | 1.0 | 1.0 |
| 30 deg | 1.0 | 1.0 | 1.0 |
| 35 deg | 1.0 | 1.0 | 1.0 |

MVDR and MUSIC resolve two targets at 15-degree separation. Bartlett requires 30-degree separation, consistent with its wider main lobe.

### SNR Sensitivity of Angle Estimators (20 trials, single target at Az = 10.33 deg, 0.2-degree grid)

This evaluates the noise sensitivity of the angle estimators using the expected range bin. Angle success rate is the fraction of trials where the estimated peak falls within 3 degrees of the true azimuth. RMSE is computed only for successful trials.

| SNR (dB) | Bartlett RMSE (deg) | MUSIC RMSE (deg) |
| :-- | :-- | :-- |
| -10 | 0.0954 | 0.0954 |
| -5 | 0.0889 | 0.0889 |
| 0 | 0.0742 | 0.0742 |
| 5 | 0.0700 | 0.0700 |
| 10 | 0.0700 | 0.0700 |
| 20 | 0.0700 | 0.0700 |

The residual RMSE at high SNR (0.07 deg) reflects the finite 0.2-degree scan grid quantization. Low-SNR RMSE increases slightly due to noise-induced peak shifts.

### Calibration (20 trials, true azimuth = 15.0 deg, random gain/phase mismatch)

| Condition | Mean Error (deg) | Std Error (deg) |
| :-- | :-- | :-- |
| Uncalibrated | 1.025 | 0.7822 |
| Calibrated | 0.0 | 0.0 |

### Runtime Benchmark (3 repeats, full pipeline per run)

| Algorithm | Mean Runtime (s) | Std (s) |
| :-- | :-- | :-- |
| Bartlett | 0.1655 | 0.0005 |
| MVDR | 0.1672 | 0.0022 |
| MUSIC | 0.2084 | 0.0150 |

## 14. Reproducibility

Fixed seeds make the simulation deterministic under comparable environments. Note that exact runtime metrics may vary across different hardware.

To reproduce from a clean clone:

```bash
git clone https://github.com/prahas12/FMCW-TDM-MIMO-Radar-3D-DOA-Estimation-and-Beamforming.git
cd FMCW-TDM-MIMO-Radar-3D-DOA-Estimation-and-Beamforming
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -e .[dev]
pytest -q
radar_pipeline run --config configs/baseline.yaml
radar_pipeline experiment --config configs/baseline.yaml
```

## 15. Configuration

All parameters are specified in `configs/baseline.yaml`. Key sections:

- `radar`: Carrier frequency, bandwidth, chirp timing, ADC settings, antenna counts.
- `array`: TX and RX element spacings in units of half-wavelengths.
- `targets`: List of targets with range, velocity, azimuth, elevation, and RCS.
- `simulation`: SNR, noise figure, random seed.
- `processing`: FFT sizes and window types.
- `detection`: CFAR type, false alarm probability, guard and training cell counts.
- `angle_estimation`: Scan grid bounds and resolution, estimation method, source enumeration method.

## 16. Limitations

- The 3TX/4RX staggered array produces a sparse virtual array with a total horizontal aperture of approximately 1.75 wavelengths. This limits the Bartlett 3-dB beamwidth to approximately 30 degrees. MVDR and MUSIC provide improved resolution but cannot overcome the fundamental aperture constraint.
- The scan grid resolution (default 1 degree in the baseline, 0.2 degrees in experiments) introduces a quantization floor on angle RMSE.
- Perfectly coherent targets (identical range and identical Doppler) produce a rank-1 covariance matrix regardless of snapshot count. Resolving such targets with MUSIC requires spatial smoothing, which is not applied in the baseline pipeline.
- The simulation models point targets with additive white Gaussian noise. Clutter, multipath, and mutual coupling are not modeled.

## 17. References

- Stoica, P., and Moses, R. L. (2005). Spectral Analysis of Signals. Pearson Prentice Hall.
- Richards, M. A. (2014). Fundamentals of Radar Signal Processing, 2nd ed. McGraw-Hill Education.

## 18. License

MIT License. See [LICENSE](LICENSE) for full text.
