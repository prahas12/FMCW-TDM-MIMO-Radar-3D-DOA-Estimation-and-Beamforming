import numpy as np

def compute_covariance(X, spatial_smoothing=False, sub_array_size=None):
    # X shape: (M, N) where M is channels, N is snapshots
    M, N = X.shape
    R = (X @ X.conj().T) / N
    
    if spatial_smoothing and sub_array_size is not None and sub_array_size < M:
        L = M - sub_array_size + 1
        R_ss = np.zeros((sub_array_size, sub_array_size), dtype=complex)
        for i in range(L):
            R_ss += R[i:i+sub_array_size, i:i+sub_array_size]
        return R_ss / L
    return R

def music_2d(R, A_mat, num_sources=1):
    M = R.shape[0]
    evals, evecs = np.linalg.eigh(R)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    
    En = evecs[:, num_sources:]
    spectrum = np.zeros((A_mat.shape[0], A_mat.shape[1]))
    
    for i in range(A_mat.shape[0]):
        for j in range(A_mat.shape[1]):
            a = A_mat[i, j, :].reshape(-1, 1)
            den = np.abs(a.conj().T @ En @ En.conj().T @ a)[0, 0]
            spectrum[i, j] = 10 * np.log10(1.0 / (den + 1e-12))
            
    return spectrum
