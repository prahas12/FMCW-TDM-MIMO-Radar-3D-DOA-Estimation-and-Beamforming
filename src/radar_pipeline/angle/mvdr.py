import numpy as np

def mvdr_2d(R, A_matrix, dl=1e-6):
    # Diagonal loading
    R_dl = R + dl * np.trace(R) * np.eye(R.shape[0])
    R_inv = np.linalg.inv(R_dl)
    
    n_el, n_az, n_ch = A_matrix.shape
    P_mvdr = np.zeros((n_el, n_az))
    
    for i in range(n_el):
        for j in range(n_az):
            a = A_matrix[i, j, :]
            denom = np.abs(a.conj().T @ R_inv @ a)
            if denom < 1e-12: denom = 1e-12
            P_mvdr[i, j] = 1.0 / denom
            
    return 10 * np.log10(P_mvdr / np.max(P_mvdr))
