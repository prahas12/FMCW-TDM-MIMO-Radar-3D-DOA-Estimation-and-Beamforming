import numpy as np

def bartlett_2d(R, A_matrix):
    n_el, n_az, n_ch = A_matrix.shape
    P_bartlett = np.zeros((n_el, n_az))
    
    for i in range(n_el):
        for j in range(n_az):
            a = A_matrix[i, j, :]
            num = np.abs(a.conj().T @ R @ a)
            P_bartlett[i, j] = num
            
    return 10 * np.log10(P_bartlett / np.max(P_bartlett))
