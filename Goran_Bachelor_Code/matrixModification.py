#In this file I attempt to generate random matrices that are pretty skewed
import numpy as np
import Throughput_as_Function as fct
def generate_doubly_stochastic_matrix(N, min_val=1e-3, max_val=0.9,alpha =0.1, beta=10.0, max_iters=10000, tol=1e-12):#Generates 1-doubly-stochastic skewed matrix of size NxN
    # Step 1: Initialize random matrix with beta distribution
    matrix = np.random.beta(alpha, beta, (N, N)) * (max_val - min_val) + min_val
    np.fill_diagonal(matrix, 0)
    print(matrix)
    
    return(Sinkhorn_Knopp(matrix))


def Sinkhorn_Knopp(matrix , max_iters=10000, tol=1e-12):
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


def add_multiplicative_noise(M, N, delta):
    matrix = np.random.normal(1,delta,(N,N))
    matrix = np.clip(matrix, 0.1, 2)
    np.fill_diagonal(matrix, 1)
    res  = M * matrix
    print(np.array2string(res,formatter={'float_kind':lambda x: "%.5f" % x}))
    return Sinkhorn_Knopp(M*matrix)
def add_additive_noise(M, N, delta):
    matrix = np.random.normal(0,delta,(N,N))
    np.fill_diagonal(matrix, 0)
    M = np.clip((M+matrix), 0, 100)
    print(np.array2string(M,formatter={'float_kind':lambda x: "%.5f" % x}))
    return Sinkhorn_Knopp(M)



if __name__ == "__main__":
    N = 16
    matrixdir="/home/studium/Documents/Code/rdcn-throughput/matrices/"
    M = demandMatrix = np.loadtxt(matrixdir+"topoopt"+".mat", usecols=range(N))




    # print(np.array2string(fct.return_normalized_matrix(mult_M),formatter={'float_kind':lambda x: "%.5f" % x}))

    # print(np.array2string(fct.return_normalized_matrix(add_M),formatter={'float_kind':lambda x: "%.5f" % x}))



    print(np.array2string(M,formatter={'float_kind':lambda x: "%.5f" % x}))

    print(np.array2string(Sinkhorn_Knopp(M),formatter={'float_kind':lambda x: "%.5f" % x}))
    print("_________________________\n\n\n\n")
    print(np.array2string(Sinkhorn_Knopp(M)-M,formatter={'float_kind':lambda x: "%.5f" % x}))