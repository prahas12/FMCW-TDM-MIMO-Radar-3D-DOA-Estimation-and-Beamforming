import numpy as np
from radar_pipeline.radar.waveform import FMCWWaveform
from radar_pipeline.radar.scene import RadarScene
from radar_pipeline.radar.mimo_virtual_array import MIMOArray

class RadarSimulator:
    def __init__(self, config):
        self.waveform = FMCWWaveform(config)
        self.scene = RadarScene(config)
        self.array = MIMOArray(config, self.waveform.lambda_c)
        self.config = config
        self.snr_db = float(config['simulation'].get('snr_db', 20.0))
        self.seed = int(config['simulation'].get('seed', 42))
        np.random.seed(self.seed)

    def generate_data_cube(self):
        num_tx = self.array.num_tx
        num_rx = self.array.num_rx
        num_v = num_tx * num_rx
        num_c = self.waveform.n_chirps
        num_s = self.waveform.n_samples
        
        data = np.zeros((num_v, num_c, num_s), dtype=complex)
        t_fast = np.arange(num_s) / self.waveform.fs
        
        for target in self.scene.targets:
            fb = (2 * self.waveform.slope * target.range) / self.waveform.c
            fd = (2 * target.velocity) / self.waveform.lambda_c
            
            az_rad = np.deg2rad(target.azimuth)
            el_rad = np.deg2rad(target.elevation)
            # Radar conventions: X is azimuth, Y is elevation, Z is broadside (depth)
            ux = np.sin(az_rad) * np.cos(el_rad)
            uy = np.sin(el_rad)
            uz = np.cos(az_rad) * np.cos(el_rad)
            u = np.array([ux, uy, uz])
            
            k = 2 * np.pi / self.waveform.lambda_c
            
            # Genuine TDM-MIMO: each Tx transmits at a different time slot
            # A full MIMO frame takes num_tx * Tc
            for c_idx in range(num_c):
                for t_idx in range(num_tx):
                    # Absolute slow time at start of this Tx chirp
                    t_slow = c_idx * (num_tx * self.waveform.t_chirp_total) + t_idx * self.waveform.t_chirp_total
                    
                    tx_phase = k * np.dot(self.array.tx_pos[t_idx], u)
                    
                    for r_idx in range(num_rx):
                        rx_phase = k * np.dot(self.array.rx_pos[r_idx], u)
                        spatial_phase = tx_phase + rx_phase
                        
                        v_idx = t_idx * num_rx + r_idx
                        phase_c = spatial_phase + target.initial_phase
                        base_phase = 2 * np.pi * (fb * t_fast + fd * t_slow) + phase_c
                        
                        signal = target.amplitude * np.exp(1j * base_phase)
                        data[v_idx, c_idx, :] += signal
                        
        signal_power = np.mean(np.abs(data)**2)
        if signal_power == 0:
            signal_power = 1.0
        noise_power = signal_power / (10**(self.snr_db / 10))
        noise = np.sqrt(noise_power / 2) * (np.random.randn(*data.shape) + 1j * np.random.randn(*data.shape))
        return data + noise
