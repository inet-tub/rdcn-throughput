#In this file I attempt to generate random matrices that are pretty skewed
import numpy as np
def generate_doubly_stochastic_matrix(N, min_val=1e-3, max_val=0.9,alpha =0.1, beta=10.0, max_iters=10000, tol=1e-12):#Generates 1-doubly-stochastic skewed matrix of size NxN
    # Step 1: Initialize random matrix with beta distribution
    matrix = np.random.beta(alpha, beta, (N, N)) * (max_val - min_val) + min_val
    np.fill_diagonal(matrix, 0)
    print(matrix)
    
    # Step 2: Sinkhorn-Knopp algorithm to iteratively normalize rows and columns
    for _ in range(max_iters):
        # Normalize rows
        row_sums = matrix.sum(axis=1)
        matrix = matrix / row_sums[:, np.newaxis]
        
        # Normalize columns
        col_sums = matrix.sum(axis=0)
        matrix = matrix / col_sums[np.newaxis, :]
        
        # Check convergence (small changes in matrix)
        if np.all(np.abs(matrix.sum(axis=1) - 1) < tol) and np.all(np.abs(matrix.sum(axis=0) - 1) < tol):
            break
    
    return matrix
if __name__ == "__main__":
    N = 8
    demand = generate_doubly_stochastic_matrix(N, min_val=1e-5,max_val=0.99)
    print(np.array2string(demand,formatter={'float_kind':lambda x: "%.5f" % x}))


    