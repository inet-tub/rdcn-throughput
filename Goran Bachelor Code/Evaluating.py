import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import numpy as np
import Throughput_as_Function as fct
import Floor_Draft as fd
import Rounding_Draft as rd
if __name__ == "__main__":
    workdir="/home/studium/Documents/Code/rdcn-throughput/matrices/"
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
    N = 16
    dE = 8
    # # floorResults =  []
    # # roundingResults = []
    CircleRes = []
    ChordRes = []
    for matrix in matrices16:
        demandMatrix = np.loadtxt(workdir+matrix+".mat", usecols=range(N))
        demand = demandMatrix *dE
        # floorResults.append(fd.thetaByFloor(N,dE, demand, 6))
        # roundingResults.append(rd.thetaByRounding(N,dE, demand, 6))
        CircleRes.append(fct.thetaEdgeFormulation(fct.createCircleGraph(N, dE), demand, N))
        ChordRes.append(fct.thetaEdgeFormulation(fct.createPseudoChord(N,dE), demand, N))

    print("Results for each matrix using Circle Graph:")
    for i in range(len(matrices16)):
        print(matrices16[i] + ": " + str(CircleRes[i]))
    print("____________________________________________________________________")
    print("Results for each matrix using Chord Graph :" )
    for i in range(len(matrices16)):
        print(matrices16[i] + ": " + str(ChordRes[i]))