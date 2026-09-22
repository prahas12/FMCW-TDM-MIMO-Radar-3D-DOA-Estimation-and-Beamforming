import numpy as np

class MIMOArray:
    def __init__(self, config, lambda_c):
        self.num_tx = int(config['radar']['tx_count'])
        self.num_rx = int(config['radar']['rx_count'])
        self.lambda_c = lambda_c
        
        self.tx_pos = np.zeros((self.num_tx, 3))
        self.rx_pos = np.zeros((self.num_rx, 3))
        
        # Planar array geometry (X=Azimuth, Y=Elevation, Z=Broadside)
        # To avoid ambiguities, we use a filled or well-defined grid.
        # Tx spacing: typically 2*lambda in X, Rx: 0.5*lambda in X, 0.5*lambda in Y
        tx_dx = float(config['array'].get('tx_spacing_x', 2.0)) * (lambda_c / 2.0)
        tx_dy = float(config['array'].get('tx_spacing_y', 0.5)) * (lambda_c / 2.0)
        rx_dx = float(config['array'].get('rx_spacing_x', 0.5)) * (lambda_c / 2.0)
        rx_dy = float(config['array'].get('rx_spacing_y', 0.5)) * (lambda_c / 2.0)
        
        # RX as a uniform linear array on X axis
        for i in range(self.num_rx):
            self.rx_pos[i, 0] = i * rx_dx
            self.rx_pos[i, 1] = 0.0
            
        # TX positions: L-shape to create 2D virtual array
        self.tx_pos[0, 0] = 0.0
        self.tx_pos[0, 1] = 0.0
        if self.num_tx > 1:
            self.tx_pos[1, 0] = tx_dx
            self.tx_pos[1, 1] = 0.0
        if self.num_tx > 2:
            self.tx_pos[2, 0] = 0.0
            self.tx_pos[2, 1] = tx_dy  # Elevation offset
            
        self.virtual_pos, self.tx_rx_map = self._compute_virtual_array()
        
    def _compute_virtual_array(self):
        v_pos = np.zeros((self.num_tx * self.num_rx, 3))
        tx_rx_map = []
        idx = 0
        for t_idx, tx in enumerate(self.tx_pos):
            for r_idx, rx in enumerate(self.rx_pos):
                v_pos[idx] = tx + rx
                tx_rx_map.append((t_idx, r_idx))
                idx += 1
        return v_pos, tx_rx_map
