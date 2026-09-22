import numpy as np

class Target:
    def __init__(self, r, v, az, el, rcs=1.0):
        self.range = r
        self.velocity = v
        self.azimuth = az
        self.elevation = el
        self.rcs = rcs
        self.amplitude = np.sqrt(rcs)
        # Random initial phase for each target to ensure they are incoherent
        self.initial_phase = np.random.uniform(0, 2*np.pi)

class RadarScene:
    def __init__(self, config):
        self.targets = []
        for t in config.get('targets', []):
            self.targets.append(Target(t['range'], t['velocity'], t['azimuth'], t['elevation'], t.get('rcs', 1.0)))
