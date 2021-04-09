from sklearn.datasets.samples_generator import make_blobs 
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.validation import as_float_array
import numpy as np

centers = [[1, 1], [-1, -1], [1, -1], [-1, -1]] 
X, labels_true = make_blobs(n_samples = 400, centers = centers,cluster_std = 0.5, random_state = 0)
s = euclidean_distances(X,squared=True) #similarity matrix
s = as_float_array(s, copy=True)

preference = np.median(s)
n_samples = s.shape[0]
# Place preference on the diagonal of S
s.flat[::(n_samples + 1)] = preference

# Initialize messages
A = np.zeros((n_samples, n_samples))
R = np.zeros((n_samples, n_samples))  
# Intermediate results

tmp = np.zeros((n_samples, n_samples))

#iteration to converge(if border doesn't change within this value)

convergence_iter = 15

# Execute parallel affinity propagation updates
e = np.zeros((n_samples, convergence_iter))

ind = np.arange(n_samples)

maxiter =100
for iteration in range(maxiter):
    # tmp = A + S; compute responsibilities
    np.add(A, s, tmp)
    I = np.argmax(tmp, axis=1)
    Y = tmp[ind, I]  # np.max(A + S, axis=1)
    tmp[ind, I] = -np.inf
    Y2 = np.max(tmp, axis=1)
    
    # tmp = Rnew
    np.subtract(s, Y[:, None], tmp)
    tmp[ind, I] = s[ind, I] - Y2
    
    
    # Damping
    damping =0.5
    tmp *= 1 - damping
    R *= damping
    R += tmp     
    
    # tmp = Rp; compute availabilities
    np.maximum(R, 0, tmp)
    tmp.flat[::n_samples + 1] = R.flat[::n_samples + 1]

    # tmp = -Anew
    tmp -= np.sum(tmp, axis=0)
    dA = np.diag(tmp).copy()
    tmp.clip(0, np.inf, tmp)
    tmp.flat[::n_samples + 1] = dA

    # Damping
    tmp *= 1 - damping
    A *= damping
    A -= tmp
        
    # Check for convergence
    E = (np.diag(A) + np.diag(R)) > 0
    e[:, iteration % convergence_iter] = E
    K = np.sum(E, axis=0) 

I = np.flatnonzero(E)
K = I.size  # Identify exemplars  

if K > 0:
        c = np.argmax(s[:, I], axis=1)
        c[I] = np.arange(K)  # Identify clusters
        # Refine the final set of exemplars and clusters and return results
        for k in range(K):
            ii = np.where(c == k)[0]
            j = np.argmax(np.sum(s[ii[:, np.newaxis], ii], axis=0))
            I[k] = ii[j]

        c = np.argmax(s[:, I], axis=1)
        c[I] = np.arange(K)
        labels = I[c]
        # Reduce labels to a sorted, gapless, list
        cluster_centers_indices = np.unique(labels)
        labels = np.searchsorted(cluster_centers_indices, labels)
        
        
        
