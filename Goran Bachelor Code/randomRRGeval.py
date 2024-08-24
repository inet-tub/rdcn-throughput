#Code to evaluate how many iterations of RRGs are needed to get the best throughput or close to the best achievable throughput
import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import numpy as np
import Throughput_as_Function as fct
import matplotlib.pyplot as plt


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

def logEveryRRGres(M, N, d, iter):
    res = []
    for i in range(iter):
        G_temp = nx.random_regular_graph(d,N)
        res.append(fct.thetaEdgeFormulation(G_temp, M, N))
    return res

def evaluateRRGLog(log):
    arr = np.array(log)
    variance = np.var(arr)
    best_index = np.argmax(arr)
    worst_index = np.argmin(arr)
    return (" | Variance: " +  str(variance) +", Worst result: " + str(arr[worst_index]) + ", Best result: "  + str(arr[best_index])+ " achieved in iteration "+ str(best_index))

def PlotLogProgression(logs, labels):
    # Set up the plot
    plt.figure(figsize=(12, 8))

    # Iterate over each list of throughput values
    for idx, throughput_values in enumerate(logs):
        x_values = []
        y_values = []
        current_best = 0
        
        # Find new best values for the current list
        for i, value in enumerate(throughput_values):
            if value > (current_best+1e-6):
                current_best = value
                x_values.append(i)
                y_values.append(value)
        
        plt.plot(x_values, y_values, 'o-', label=labels[idx])

    plt.xlabel("Iteration")
    plt.ylabel("Throughput")
    plt.title("Best Throughput by Iteration for different M with RRG variation")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    workdir="/home/studium/Documents/Code/rdcn-throughput/matrices/"
    N = 16
    dE = 8
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
    evals = []
    logs = []
    for matrix in matrices16:
        demandMatrix = np.loadtxt(workdir+matrix+".mat", usecols=range(N))
        demand = demandMatrix *dE
        log = logEveryRRGres(demand, N, dE, iter=100)
        logs.append(log)
        evals.append(matrix + evaluateRRGLog(log))
        # matrix_progress.append(findBestRRGwithLog(demand, N, dE, iter=100))  #Takes ~1h even for N=16 if you set iter to 100 on my laptop
    PlotLogProgression(logs, matrices16)
    for eval in evals:
        print(eval)
