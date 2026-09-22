import numpy as np
from scipy import constants

class FMCWWaveform:
    def __init__(self, config):
        self.fc = float(config['radar']['carrier_frequency'])
        self.bw = float(config['radar']['bandwidth'])
        self.tc = float(config['radar']['chirp_duration'])
        self.idle = float(config['radar']['idle_time'])
        self.fs = float(config['radar']['adc_sample_rate'])
        self.n_samples = int(config['radar']['adc_samples'])
        self.n_chirps = int(config['radar']['num_chirps'])
        
        self.c = constants.c
        self.lambda_c = self.c / self.fc
        self.slope = self.bw / self.tc
        self.t_chirp_total = self.tc + self.idle
        self.t_frame = self.t_chirp_total * self.n_chirps
        
        self.range_res = self.c / (2 * self.bw)
        self.max_range = (self.fs / 2) * self.c / (2 * self.slope)
        self.vel_res = self.lambda_c / (2 * self.t_frame)
        self.max_vel = self.lambda_c / (4 * self.t_chirp_total)
