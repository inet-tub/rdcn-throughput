#Code to evaluate how many iterations of RRGs are needed to get the best throughput or close to the best achievable throughput
import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import numpy as np
import Throughput_as_Function as fct


workdir="/home/studium/Documents/Code/rdcn-throughput/matrices/"
N = 16
dE = 4

def findBestRRGwithLog(M, N, d, iter): #Given demand Matrix M, test out RRGs for given nr. of iterations return log of tuples: best troughput values so far in what iteration
    best_iters = []
    k = 0
    for i in range(iter):
        G_temp = nx.random_regular_graph(d,N)
        theta = fct.thetaEdgeFormulation(G_temp, M, N)
        if(k ==0 or theta > (best_iters[k-1][1]+1e-6)):#We don't care about improvements smaller than 1e-6
            best_iters.append((i, theta))
            k +=1
    return(best_iters)


matrices16 = [
            "chessboard-16",
            "uniform-16",
            "permutation-16",
            "skew-16-0.0",
            "skew-16-0.1",
            "skew-16-0.2",
            "skew-16-0.3",
            "skew-16-0.4",
            "skew-16-0.5",
            "skew-16-0.6",
            "skew-16-0.7",
            "skew-16-0.8",
            "skew-16-0.9",
            "skew-16-1.0",
            "data-parallelism","hybrid-parallelism","heatmap2","heatmap3"]
matrix_progress = []
for matrix in matrices16:
    demandMatrix = np.loadtxt(workdir+matrix+".mat", usecols=range(N))
    demand = demandMatrix *dE
    matrix_progress.append(findBestRRGwithLog(demand, N, dE, iter=100))  #Takes ~1h even for N=16 if you set iter to 100 on my laptop
for entry in matrix_progress:
    print(entry)#Goes through every demand matrix and prints the log for each of them.
