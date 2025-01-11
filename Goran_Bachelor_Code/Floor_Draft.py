#This file contains a first rough draft of the Rounding heuristic
import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import numpy as np
import Throughput_as_Function as fct
import matplotlib.pyplot as plt
import pandas as pd
import matrixModification as mm
def OneFloorIter(N, d, M, iteration):#Returns throughput that can be achieved for given N, d and M with floor heuristic. RRGiter determines how many RRGs you will try per iter
    # print(M)
    if(iteration != 1.00):
        demand = M*iteration #scale down M (demandMatrix) by factor theta (iteration)
    else: 
        demand = np.array(M)
    # np.savetxt('demandBefore.txt', demand)
    # print(np.array2string(demand,formatter={'float_kind':lambda x: "%.5f" % x}))
    linkCapacity = np.floor(demand)
    # print(linkCapacity)
    # np.savetxt('demandFloor.txt', linkCapacity)


    outDegree = [0]*N # How many outgoing links each node has left
    inDegree = [0]*N # How many incoming links each node has left
    for row in range(N):
        outDegree[row] = int(np.min([d,N-1]) - np.sum(linkCapacity[row]))
    for column in range(N):
        inDegree[column] = int(np.min([d,N-1]) - np.sum(linkCapacity[:,column]))

    # print(outDegree)
    # print(inDegree)
    dRes = np.min([N-1, np.min([outDegree,inDegree])])#Smallest val 


    outLeft = [ x - dRes for x in outDegree]
    inLeft = [ x - dRes for x in inDegree]
    # print(outLeft)
    # print(inLeft)
    
    G = nx.random_regular_graph(dRes,N)
    fct.addGraphToMatrix(G, linkCapacity)
    fct.match_and_increment(outLeft, inLeft, linkCapacity)
    # print(linkCapacity)
    res = fct.thetaEdgeFormulation(linkCapacity, M, N, input_graph=False)
    return res


def ThetaByFloor(N, d, M, measure_SH = False):#Returns throughput that can be achieved for given N, d and M with floor heuristic. RRGiter determines how many RRGs you will try per iter
    iterations=[1-i*0.01 for i in range(99)]
    for iteration in iterations:


    
        if(iteration != 1.00):
            demand = M*iteration #scale down M (demandMatrix) by factor theta (iteration)
        else: 
            demand = np.array(M)
        linkCapacity = np.floor(demand) #FloorMatrix

        outDegree = [0]*N # How many outgoing links each node has left
        inDegree = [0]*N # How many incoming links each node has left
        for row in range(N):
            outDegree[row] = int(np.min([d,N-1]) - np.sum(linkCapacity[row]))
        for column in range(N):
            inDegree[column] = int(np.min([d,N-1]) - np.sum(linkCapacity[:,column]))
        dRes = np.min([N-1, np.min([outDegree,inDegree])])#Smallest val 
        outLeft = [ x - dRes for x in outDegree]
        inLeft = [ x - dRes for x in inDegree]

        G = nx.random_regular_graph(dRes,N)
        fct.addGraphToMatrix(G, linkCapacity)
        fct.match_and_increment(outLeft, inLeft, linkCapacity)
        if(measure_SH):
            res, routed = fct.thetaEdgeFormulation(linkCapacity, demand, N, measure_SH=True, input_graph=False)
            ratioSH = routed[1] / routed[0]
            if(res ==1):
                return (iteration, ratioSH)
        else:
            res = fct.thetaEdgeFormulation(linkCapacity, demand, N, measure_SH=False, input_graph=False)
        if res== 1:
            return iteration
    return 0
if __name__ == "__main__":
    matrixname= "skew-16-0.2"
    N= 16
    dE = 14
    workdir="/home/studium/Documents/Code/rdcn-throughput/matrices/"
    demandMatrix = np.loadtxt(workdir+matrixname+".mat", usecols=range(N))
    demandMatrix = mm.Sinkhorn_Knopp(demandMatrix)
    # fct.filtering(demandMatrix)
    # demandMatrix = fct.return_normalized_matrix(demandMatrix)
    demandMatrix = demandMatrix * dE
    # start = time.time()


    # plotGamma(8,dE, demandMatrix)

    # findThroughput(16,14,matrixname, "Optimal")

    print(fct.plotGamma(N, dE, demandMatrix, matrixname))

    # print(OneFloorIter(8, dE, demandMatrix, 0.7))

    # end = time.time()
    # length = end - start
    # print(length)
    # plotGamma(N, dE, demandMatrix)

