#This file contains a first rough draft of the Rounding heuristic
import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import numpy as np
import Throughput_as_Function as fct
import paperCode as pc
def rounding(M, N, d):#Given M, N and d returns rounded numpy matrix sol such that sum of all rows and all columns of sol equal to function parameter d; Assumes d-doubly stochastic matrix 
    model = gp.Model()
    entry_vars = {}
    print(np.array2string(M))
    print(d)
    model.params.LogToConsole = 0
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
    # print(np.array2string(sol))
    return sol
def createResidual(M, integer): #Modifies demand matrix M such that it becomes the residual matrix
    N = M.shape[0]
    for i in range(N):
        for j in range(N):
            if i !=j:
                M[i,j] = M[i,j]-round(integer[i,j]) #No np.max with 0, we can use negative value to determine how much capacity direct edge btwn i and j has left in rounding heuristic
def residualRounding(M, N, rowGoals, colGoals):
    model = gp.Model()
    model.Params.LogToConsole = 0
    entry_vars = {}
    row_sum_vars = {}
    col_sum_vars = {}
    #Every entry in demand matrix has integer var that is either floor or ceiling
    for i in range(N):
        for j in range(N):
            if i !=j:
                entry_vars[i,j] = model.addVar(vtype=GRB.INTEGER,name=f"entry_{i}_{j}", lb=0, ub= 1)
    # for i in range(N):
    #     row_sum_vars[i] = model.addVar(vtype=GRB.INTEGER,name=f"row_sum_{i}", lb=rowGoals[i]-1, ub = rowGoals[i])
    #     col_sum_vars[i] =  model.addVar(vtype=GRB.INTEGER,name=f"col_sum_{i}", lb=colGoals[i]-1, ub = colGoals[i])

    #Add row and col constraints
    for i in range(N):
        model.addConstr(gp.quicksum(entry_vars[i,j] for j in range(N) if j !=i) == rowGoals[i]) # Row Sum
        model.addConstr(gp.quicksum(entry_vars[j,i] for j in range(N) if j != i) == colGoals[i]) # Col Sum

    model.update()
    diff = model.addVar(vtype=GRB.CONTINUOUS, lb=-1e10 ,ub=1e10) 
    model.addConstr(gp.quicksum(((entry_vars[i,j] - M[i,j])*entry_vars[i,j]) for i in range(N) for j in range(N) if j != i)== diff)
    model.setObjective(diff, GRB.MINIMIZE) #Since this is purely a feasibility problem we optimize a constant of 0
    
    # Optimize the model
    model.optimize()
    #Create Rounded Matrix from gurobi variable values and return reference to that matrix
    sol = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i !=j:
                sol[i,j] = entry_vars[i,j].X
    # print(np.array2string(sol)) #Debug rounding solution print statement
    return sol
def thetaByRounding(N, d, M, RRGiter):#Returns throughput that can be achieved for given N, d and M with rounding heuristic. RRGiter determines how many RRGs you will try per iter
    iterations=[1-i*0.01 for i in range(99)]
    for iteration in iterations:
        print("################")
        print(iteration)
        print("################")
        demand = M*iteration #scale down M (demandMatrix) by factor theta (iteration)
        print(np.array2string(demand))
        # floor = np.floor(demand)
        
        dBulk = np.floor(d*iteration).astype(int) #dIter describes how much of the edge constraint d is reserved for meeting bulk demand in rounding phase
        dRes = d - dBulk #dRes describes how much is left to construct a dRes-RRG to meet residual demand
        roundedMatrix = rounding(demand, N, dBulk)
        createResidual(demand, roundedMatrix)  #
        print(np.array2string(demand, formatter={'float_kind':lambda x: "%.3f" % x}))
        # G = nx.random_regular_graph(dRes, N)
        (_,res, _) = fct.findBestRRG(demand, N, dRes,RRGiter)
        if res == 1:
            return iteration
    return 0

def alternativeTheta(N, d, M, RRGiter):
    iterations=[1-i*0.01 for i in range(99)]
    for iteration in iterations:
        # print("################")
        # print(iteration)
        # print("################")
        demand = M*iteration #scale down M (demandMatrix) by factor theta (iteration)
        # print(np.array2string(demand,formatter={'float_kind':lambda x: "%.2f" % x}))
        # print("___________________________")
        dBulk = np.floor(d*iteration).astype(int) #dIter describes how much of the edge constraint d is reserved for meeting bulk demand in rounding phase
        dRes = d - dBulk #dRes describes how much is left to construct a dRes-RRG to meet residual demand
        
        floor = np.floor(demand)
        # print(np.array2string(floor,formatter={'float_kind':lambda x: "%.1f" % x}))
        # print("___________________________")


        createResidual(demand, floor) 
        # print(np.array2string(demand,formatter={'float_kind':lambda x: "%.5f" % x}))
        # print("___________________________")
        floorRowSums = floor.sum(axis=1)
        floorColSums = floor.sum(axis=0)
        # print(floorRowSums) 
        # print(floorColSums) 
        floorRowRemainders= dBulk - floorRowSums
        floorColRemainders =  dBulk - floorColSums
        # print(floorRowRemainders) 
        # print(floorColRemainders) 
        # print(demand.sum(axis=1))
        # print(demand.sum(axis=0))
        # print(d*iteration)
        # print(dBulk)
        roundedMatrix = residualRounding(demand,N, floorRowRemainders, floorColRemainders)
        # print(np.array2string(roundedMatrix))
        createResidual(demand, roundedMatrix)
        # print(np.array2string(demand, formatter={'float_kind':lambda x: "%.3f" % x}))
        G = nx.random_regular_graph(dRes, N)
        # (_,res, _) = fct.findBestRRG(demand, N, dRes,RRGiter)
        res = fct.thetaEdgeFormulation(G, demand, N)
        if res == 1:
            return iteration
    return 0
def altThetawGCM(N, d, M , RRGIter): # TODO: Change this into what Vamsi meant... Round up or down and do GCM on rest 
    iterations=[1-i*0.01 for i in range(99)]
    for iteration in iterations:
        # print("################")
        # print(iteration)
        # print("################")
        demand = M*iteration #scale down M (demandMatrix) by factor theta (iteration)
        # print(np.array2string(demand,formatter={'float_kind':lambda x: "%.2f" % x}))
        # print("___________________________")
        # dBulk = np.ceil(d*iteration).astype(int) #dIter describes how much of the edge constraint d is reserved for meeting bulk demand in rounding phase
        # dRes = d - dBulk #dRes describes how much is left to construct a dRes-RRG to meet residual demand
        
        floor = np.floor(demand)
        # print(np.array2string(floor,formatter={'float_kind':lambda x: "%.1f" % x}))
        # print("___________________________")


        createResidual(demand, floor) 
        # print(np.array2string(demand,formatter={'float_kind':lambda x: "%.5f" % x}))
        # print("___________________________")
        floorRowSums = floor.sum(axis=1)
        floorColSums = floor.sum(axis=0)
        # print("Floor Sums")
        # print(floorRowSums) 
        # print(floorColSums) 
        BulkRowRemainders= d - floorRowSums
        BulkColRemainders =  d - floorColSums
        # print("Bulk remainders")
        # print(BulkRowRemainders) 
        # print(BulkColRemainders) 
        # print(demand.sum(axis=1))
        # print(demand.sum(axis=0))
        # print(d*iteration)
        # print(dBulk)
        # roundedMatrix = residualRounding(demand,N, floorRowRemainders, floorColRemainders)
        roundedMatrix = generalized_roundingwObj(demand, N)
        
        roundedRowSums = roundedMatrix.sum(axis=1)
        roundedColSums = roundedMatrix.sum(axis=0)
        # print("Rounded Sums")
        # print(roundedRowSums)
        # print(roundedColSums)
        BulkRowRemainders-=roundedRowSums
        BulkColRemainders-= roundedColSums
        # print("Bulk remainders")
        # print(BulkRowRemainders)
        # print(BulkColRemainders)
        BulkRowRemainders = BulkRowRemainders.astype(int)
        BulkColRemainders = BulkColRemainders.astype(int)
        dRes = np.min([N-1, np.min([BulkRowRemainders,BulkColRemainders])])
        BulkRowRemainders-=dRes
        BulkColRemainders-= dRes
        #Row Out; Col In
        # GCMOfRest = nx.directed_configuration_model(BulkColRemainders.tolist(), BulkRowRemainders.tolist())
        fct.match_and_decrement(BulkColRemainders, BulkColRemainders, demand)
        # for i in range(N):
        #     for j in range(N):
        #         if(GCMOfRest.number_of_edges(i,j) != 0):
        #             print(str(i) + " " + str(j) + ": " +str(GCMOfRest.number_of_edges(i,j)))
        #         demand[i,j] -= GCMOfRest.number_of_edges(i,j)
        createResidual(demand, roundedMatrix)
        G = nx.random_regular_graph(dRes, N)
        # (_,res, _) = fct.findBestRRG(demand, N, dRes,RRGIter)
        res = fct.thetaEdgeFormulation(G, demand, N)
        if res == 1:
            return iteration
    return 0
def generalized_roundingwObj(M, N):#Given M, N and d returns rounded numpy matrix sol such that sum of all rows and all columns of sol equal to function parameter d; Assumes d-doubly stochastic matrix 
    model = gp.Model()
    model.Params.LogToConsole = 0
    entry_vars = {}
    row_sums = {}
    col_sums = {}
    row_sum_vars = {}
    col_sum_vars = {}
    #Every entry in demand matrix has integer var that is either floor or ceiling
    for i in range(N):
        for j in range(N):
            if i !=j:
                entry_vars[i,j] = model.addVar(vtype=GRB.INTEGER,name=f"entry_{i}_{j}", lb=np.floor(M[i,j]), ub= np.ceil(M[i,j]))
    for i in range(N):
        row_sums[i] = pc.trim_floats(np.sum(M[i,:])) #Necessary to trim sum floats, because otherwise you get 16.000000000004 which can be rounded up -> Not intended
        col_sums[i] = pc.trim_floats(np.sum(M[:,i]))
        row_sum_vars[i] = model.addVar(vtype=GRB.INTEGER,name=f"row_sum_{i}", lb=np.floor(row_sums[i]), ub = np.ceil(row_sums[i]))
        col_sum_vars[i] =  model.addVar(vtype=GRB.INTEGER,name=f"col_sum_{i}", lb=np.floor(col_sums[i]), ub = np.ceil(col_sums[i]))

    #Add row and col constraints
    for i in range(N):
        model.addConstr(gp.quicksum(entry_vars[i,j] for j in range(N) if j !=i) == row_sum_vars[i]) # Row Sum
        model.addConstr(gp.quicksum(entry_vars[j,i] for j in range(N) if j != i) == col_sum_vars[i]) # Col Sum

    model.update()
    diff = model.addVar(vtype=GRB.CONTINUOUS, lb=-1e10 ,ub=1e10) 
    model.addConstr(gp.quicksum(((entry_vars[i,j] - M[i,j])*entry_vars[i,j]) for i in range(N) for j in range(N) if j != i)== diff)
    model.setObjective(diff, GRB.MINIMIZE) #Since this is purely a feasibility problem we optimize a constant of 0
    # Optimize the model
    model.optimize()
    #Create Rounded Matrix from gurobi variable values and return reference to that matrix
    sol = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i !=j:
                sol[i,j] = entry_vars[i,j].X
    # print(np.array2string(sol)) #Debug rounding solution print statement
    return sol


#3 Main Approaches: Conventional : thetabyRound; alternativeTheta; and altThetawGCM each with own rounding method


# if __name__ == "__main__":
#     N= 16
#     dE = 8
#     workdir="/home/studium/Documents/Code/rdcn-throughput/matrices/"
#     demandMatrix = np.loadtxt(workdir+"random-skewed-16.mat", usecols=range(N))
#     fct.filtering(demandMatrix)
#     demandMatrix = demandMatrix * dE
#     print(altThetawGCM(N, dE, demandMatrix, 6))

