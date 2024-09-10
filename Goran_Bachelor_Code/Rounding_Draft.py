#This file contains a first rough draft of the Rounding heuristic
import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import numpy as np
import Throughput_as_Function as fct

def rounding(M, N, d):#Given M, N and d returns rounded numpy matrix sol such that sum of all rows and all columns of sol equal to function parameter d; Assumes d-doubly stochastic matrix 
    model = gp.Model()
    entry_vars = {}
    print(np.array2string(M))
    print(d)
    #Every entry in demand matrix has integer var that is either floor or ceiling
    for i in range(N):
        for j in range(N):
            if i !=j:
                entry_vars[i,j] = model.addVar(vtype=GRB.INTEGER,name=f"entry_{i}_{j}", lb=np.floor(M[i,j]), ub= np.ceil(M[i,j]))

    #Add row and col constraints
    for i in range(N):
        model.addConstr(gp.quicksum(entry_vars[i,j] for j in range(N) if j !=i) == d)
        model.addConstr(gp.quicksum(entry_vars[j,i] for j in range(N) if j != i) == d)

    model.update()
    const = model.addVar(vtype=GRB.INTEGER, ub=0)
    # diff = model.addVar(vtype=GRB.CONTINUOUS, lb=-1e10 ,ub=1e10) 
    # model.addConstr(gp.quicksum(((entry_vars[i,j] - M[i,j])*entry_vars[i,j]) for i in range(N) for j in range(N) if j != i)== diff)
    #Justification for objective function, which is useful in this case?
    model.setObjective(const, GRB.MAXIMIZE) #Since this is purely a feasibility problem we optimize a constant of 0
    
    # Optimize the model
    model.optimize()
    #Create Rounded Matrix from gurobi variable values and return reference to that matrix
    sol = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i !=j:
                sol[i,j] = entry_vars[i,j].X
    print(np.array2string(sol))
    return sol
def createResidual(M, integer): #Modifies demand matrix M such that it becomes the residual matrix
    N = M.shape[0]
    for i in range(N):
        for j in range(N):
            if i !=j:
                M[i,j] = M[i,j]-round(integer[i,j]) #No np.max with 0, we can use negative value to determine how much capacity direct edge btwn i and j has left in rounding heuristic
                
def thetaByRounding(N, d, M, RRGiter):#Returns throughput that can be achieved for given N, d and M with rounding heuristic. RRGiter determines how many RRGs you will try per iter
    iterations=[1-i*0.01 for i in range(99)]
    for iteration in iterations:
        print("################")
        print(iteration)
        print("################")
        demand = M*iteration #scale down M (demandMatrix) by factor theta (iteration)
        print(np.array2string(demand))
        floor = np.floor(demand)
        
        dBulk = np.floor(d*iteration).astype(int) #dIter describes how much of the edge constraint d is reserved for meeting bulk demand in rounding phase
        dRes = d - dBulk #dRes describes how much is left to construct a dRes-RRG to meet residual demand
        createResidual(demand, rounding(demand, N, dBulk))  #
        # G = nx.random_regular_graph(dRes, N)
        (_,res, _) = fct.findBestRRG(demand, N, dRes,RRGiter)
        if res == 1:
            return iteration
    return 0
if __name__ == "__main__":
    N= 16
    dE = 8
    workdir="/home/studium/Documents/Code/rdcn-throughput/matrices/"
    demandMatrix = np.loadtxt(workdir+"heatmap2.mat", usecols=range(N))
    demandMatrix = demandMatrix * dE
    print(str(thetaByRounding(N, dE, demandMatrix, 1)))

