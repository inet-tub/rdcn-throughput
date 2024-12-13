#This file contains a first rough draft of the Rounding heuristic
import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import numpy as np
import Throughput_as_Function as fct

def thetaByFloor(N, d, M, RRGiter, measure_SH = False):#Returns throughput that can be achieved for given N, d and M with floor heuristic. RRGiter determines how many RRGs you will try per iter
    iterations=[1-i*0.01 for i in range(99)]
    for iteration in iterations:

        if(iteration != 1.00):
            demand = M*iteration #scale down M (demandMatrix) by factor theta (iteration)
        else: 
            demand = np.array(M)
        demandFloor = np.floor(demand)

        total_demand = 0
        total_demand += demand.sum()
        SH_demand = 0
        SH_demand += demandFloor.sum()

        print("Total Demand : " + str(total_demand)+", Floor Demand: " + str(SH_demand))
        outDegree = [0]*N # How many outgoing links each node has left
        inDegree = [0]*N # How many incoming links each node has left
        for row in range(N):
            outDegree[row] = int(np.min([d,N-1]) - np.sum(demandFloor[row]))
        for column in range(N):
            inDegree[column] = int(np.min([d,N-1]) - np.sum(demandFloor[:,column]))
        dRes = np.min([N-1, np.min([outDegree,inDegree])])#Smallest val 


        outLeft = [ x - dRes for x in outDegree]
        inLeft = [ x - dRes for x in inDegree]
        # G = nx.directed_configuration_model(inDegree, outDegree)
        
        
        G = nx.random_regular_graph(dRes,N)
        fct.createResidual(demand, demandFloor)  

        # print(np.array2string(demand))
        routed = fct.match_and_decrement(outLeft, inLeft, demand)
        print(routed)
        SH_demand += routed
        # print(np.array2string(demand))
        # (_,res, _) = fct.findBestRRG(demand, N, dRes,RRGiter, cutoff= True)
        if(measure_SH):
            res, routed = fct.thetaEdgeFormulation(G, demand, N, measure_SH=True)
            SH_demand += routed[1]
            ratioSH = SH_demand / total_demand
            if(res ==1):
                return (iteration, ratioSH)
        else:
            res = fct.thetaEdgeFormulation(G, demand, N)
        if res== 1:
            return iteration
    return 0
if __name__ == "__main__":

    N= 8
    dE = 6
    workdir="/home/studium/Documents/Code/rdcn-throughput/matrices/"
    demandMatrix = np.loadtxt(workdir+"uniform-8.mat", usecols=range(8))
    fct.filtering(demandMatrix)
    demandMatrix = fct.return_normalized_matrix(demandMatrix)
    demandMatrix = demandMatrix * dE
    print(str(thetaByFloor(N, dE, demandMatrix, 6, measure_SH=True)))

