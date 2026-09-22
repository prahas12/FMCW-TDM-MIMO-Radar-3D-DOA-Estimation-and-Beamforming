import numpy as np

def compute_steering_vector(az_deg, el_deg, v_pos, lambda_c):
    az_rad = np.deg2rad(az_deg)
    el_rad = np.deg2rad(el_deg)
    
    ux = np.sin(az_rad) * np.cos(el_rad)
    uy = np.sin(el_rad)
    uz = np.cos(az_rad) * np.cos(el_rad)
    u = np.array([ux, uy, uz])
    
    k = 2 * np.pi / lambda_c
    phase = k * np.dot(v_pos, u)
    return np.exp(1j * phase)

def compute_steering_matrix(az_grid, el_grid, v_pos, lambda_c):
    A = []
    for el in el_grid:
        row = []
        for az in az_grid:
            a = compute_steering_vector(az, el, v_pos, lambda_c)
            row.append(a)
        A.append(row)
    return np.array(A) # (el, az, channels)
