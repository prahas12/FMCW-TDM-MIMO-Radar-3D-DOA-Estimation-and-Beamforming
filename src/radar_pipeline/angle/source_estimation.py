import numpy as np

def estimate_num_sources(R, method='mdl', num_snapshots=1):
    """
    Estimates the number of sources using Minimum Description Length (MDL) or AIC.
    R: Spatial covariance matrix
    """
    M = R.shape[0]
    eigenvalues = np.real(np.linalg.eigvals(R))
    # Sort descending
    eigenvalues = np.sort(eigenvalues)[::-1]
    
    # Ensure strictly positive eigenvalues for log
    eigenvalues[eigenvalues <= 1e-12] = 1e-12
    
    N = num_snapshots
    
    mdl = np.zeros(M)
    for k in range(M - 1):
        # geometric mean of remaining noise eigenvalues
        noise_eig = eigenvalues[k:]
        L = M - k
        g_mean = np.exp(np.mean(np.log(noise_eig)))
        a_mean = np.mean(noise_eig)
        
        if a_mean == 0 or g_mean / a_mean <= 0:
            mdl[k] = np.inf
            continue
            
        # Log-likelihood function
        L_k = N * L * np.log(a_mean / g_mean)
        
        # Penalty term
        if method == 'mdl':
            penalty = 0.5 * k * (2 * M - k) * np.log(N)
        elif method == 'aic':
            penalty = k * (2 * M - k)
        else:
            penalty = 0
            
        mdl[k] = L_k + penalty
        
    mdl[-1] = np.inf  # K=M-1 is the maximum allowed
    
    # K is the index that minimizes MDL
    estimated_k = np.argmin(mdl)
    
    # Fallback if 0 sources detected but CFAR triggered (implies at least 1)
    if estimated_k == 0:
        estimated_k = 1
        
    return estimated_k
