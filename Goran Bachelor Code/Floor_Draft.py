#This file contains a first rough draft of the Rounding heuristic
import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import numpy as np
import Throughput_as_Function as fct


def createResidual(M, floored): #Modifies demand matrix M such that it becomes the residual matrix
    for i in range(N):
        for j in range(N):
            if i !=j:
                M[i,j] = M[i,j]-floored[i,j]

def thetaByFloor(N, d, M, RRGiter):#Returns throughput that can be achieved for given N, d and M with floor heuristic. RRGiter determines how many RRGs you will try per iter
    iterations=[1-i*0.01 for i in range(99)]
    for iteration in iterations:
        print("################")
        print(iteration)
        print("################")
        demand = M*iteration #scale down M (demandMatrix) by factor theta (iteration)
        demandFloor = np.floor(demand)
        outDegree = [0]*N
        inDegree = [0]*N
        for row in range(N):
            outDegree[row] = int(np.min([d,N-1]) - np.sum(demandFloor[row]))
        for column in range(N):
            inDegree[column] = int(np.min([d,N-1]) - np.sum(demandFloor[:,column]))
        dRes = np.min([N-1, np.min([outDegree,inDegree])])
        # if dRes>=1:
        #     G = nx.random_regular_graph(dRes,N)
        createResidual(demand, demandFloor)  
        (_,res, _) = fct.findBestRRG(demand, N, dRes,RRGiter)
        if res== 1:
            return iteration
    return 0

N= 16
dE = 8
workdir="/home/studium/Documents/Code/rdcn-throughput/matrices/"
demandMatrix = np.loadtxt(workdir+"heatmap2.mat", usecols=range(N))
demandMatrix = demandMatrix * dE
print(str(thetaByFloor(N, dE, demandMatrix, 6)))

