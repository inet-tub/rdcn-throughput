import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import numpy as np
import Throughput_as_Function as fct
import Rounding_Draft as rd
import Floor_Draft as fd
import matplotlib.pyplot as plt

#%%
def generate_synthmatrix_names(N):
    res = [
    "chessboard-",
    "uniform-",
    "permutation-",
    "random-skewed-",
    "skew-",
    "skew-",
    "skew-",
    "skew-",
    "skew-",
    "skew-",
    "skew-",
    "skew-",
    "skew-",
    ]
    for i in range(4):
        res[i] += str(N)
    for j in range(9):
        res[j+4] += str(N) + "-0." + str(j+1)
    return res
organicmatrices16 = ["data-parallelism","hybrid-parallelism","heatmap1","heatmap2","heatmap3", "topoopt"]
workdir="/home/studium/Documents/Code/rdcn-throughput/matrices/"

def trim_floats(val, tolerance=1e-9):
    if abs(val - round(val)) < tolerance:
        return round(val)
    return val
def thetaSingleHop(G, M, N, input_graph = True):#Given static topology G and demand matrix M, returns best throughput achievable with single hops only
    model = gp.Model()
    capacity = {}
    model.Params.LogToConsole = 0
    # model.Params.OptimalityTol = 1e-9
    # model.Params.FeasibilityTol = 1e-9
    # model.Params.IntFeasTol = 1e-9
    if input_graph:
        for i in range(N):
            for j in range(N):
                    capacity[(i,j)] = G.number_of_edges(i,j) 
                    if(M[i,j]< 0): #In case of rounding, negative demand tells us how much capacity is left on that edge
                        capacity[(i, j)]-= M[i,j]
    else:
        for i in range(N):
            for j in range(N):
                if i !=j:
                    capacity[(i, j)] = G[i,j]
                if(M[i,j]< 0): #In case of rounding, negative demand tells us how much capacity is left on that edge
                        capacity[(i, j)]-= M[i,j]

    
    flow_variables = {}
    for i in range(N):
        for j in range(N):
            if i != j and capacity[(i, j)] > 0 and M[i,j] > 0:
                # Create a flow variable for each node pair (i, j) corresponding to node pairs (s, d)
                flow_variables[(i, j)] = model.addVar(vtype=GRB.CONTINUOUS, name=f'flow_{i}_{j}', lb=0)
    throughput = model.addVar(vtype=GRB.CONTINUOUS, name='throughput', lb=0, ub=1)
    model.update()
    # Implement throughput constraint
    for i in range(N):
        for j in range(N):
            if i != j and capacity[(i, j)] > 0 and M[i,j] > 0:
                # Define the throughput constraint
                throughput_constraint_expr = flow_variables[(i,j)] >= throughput * M[i, j]
                model.addConstr(throughput_constraint_expr, f'throughput_constraint_{i}_{j}')
                # Define the capacity constraint
                capacity_constraint_expr = flow_variables[(i, j)] <= capacity[(i, j)]
                model.addConstr(capacity_constraint_expr, f'capacity_constraint_{i}_{j}')
    # Set the objective to maximize throughput
    model.setObjective(throughput, GRB.MAXIMIZE)
    # Optimize the model
    model.optimize()
    return throughput.X

def thetaSingleHop2(G, M, N, input_graph = True):#Given static topology G and demand matrix M, returns best throughput achievable with single hops only
    capacity = np.zeros((N,N))
    if(input_graph):
        for i in range(N):
            for j in range(N):
                capacity[(i,j)] = G.number_of_edges(i,j) 
                if(M[i,j]< 0): #In case of rounding, negative demand tells us how much capacity is left on that edge
                        capacity[(i, j)]-= M[i,j]
    else:
        for i in range(N):
            for j in range(N):
                if i !=j:
                    capacity[(i, j)] = G[i,j]
                if(M[i,j]< 0): #In case of rounding, negative demand tells us how much capacity is left on that edge
                        capacity[(i, j)]-= M[i,j]
    
    # To avoid division by zero, handle elements where demand is zero
    # We set the ratio as infinity where demand is zero (so we ignore those cases).
    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = np.where(M != 0, capacity / M, np.inf)
    
    # Find the smallest ratio (ignoring infinities)
    smallest = min(1, np.min(ratios))
    
    return smallest


def generalized_rounding(M, N, d):#Given M, N and d returns rounded numpy matrix sol such that sum of all rows and all columns of sol equal to function parameter d; Assumes d-doubly stochastic matrix 
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
        row_sums[i] = trim_floats(np.sum(M[i,:])) #Necessary to trim sum floats, because otherwise you get 16.000000000004 which can be rounded up -> Not intended
        col_sums[i] = trim_floats(np.sum(M[:,i]))
        row_sum_vars[i] = model.addVar(vtype=GRB.INTEGER,name=f"row_sum_{i}", lb=np.floor(row_sums[i]), ub = np.ceil(row_sums[i]))
        col_sum_vars[i] =  model.addVar(vtype=GRB.INTEGER,name=f"col_sum_{i}", lb=np.floor(col_sums[i]), ub = np.ceil(col_sums[i]))

    #Add row and col constraints
    for i in range(N):
        model.addConstr(gp.quicksum(entry_vars[i,j] for j in range(N) if j !=i) == row_sum_vars[i]) # Row Sum
        model.addConstr(gp.quicksum(entry_vars[j,i] for j in range(N) if j != i) == col_sum_vars[i]) # Col Sum

    model.update()
    const = model.addVar(vtype=GRB.INTEGER, ub=0)#In this case, Vamsi mentioned that objective is unnecessary
    model.setObjective(const, GRB.MAXIMIZE) #Since this is purely a feasibility problem we optimize a constant of 0
    
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
def return_normalized_matrix(M): #Normalizes a matrix by dividing it by the scalar which is the biggest row or col sum; Afterwards every row/col sum leq 1 
    max_row_sum = M.sum(axis=1).max()
    max_col_sum = M.sum(axis=0).max()
    max_sum = max(max_row_sum, max_col_sum)
    M = np.divide(M, max_sum)
    return M # call-by-reference doesn't work for some reason, hence returning M
def randomized_vermillion_throughput(saturated_demand, d, k, N, MH = True):
    normalized_demand = return_normalized_matrix(saturated_demand)
    modifiedM = addNoisetoMatrix(saturated_demand)
    deg = ((k-1)*N)
    scaled_demand = normalized_demand * deg
    

    rounded_matrix = generalized_rounding(scaled_demand, N, deg)
    out_Left  = []
    in_Left = []
    for row in range(N):
        out_Left.append(int(deg - np.sum(rounded_matrix[row,:])))
    for col in range(N):
        in_Left.append(int(deg - np.sum(rounded_matrix[:,col])))
    # print(out_Left)
    # print(in_Left)
    G = nx.directed_configuration_model(in_Left, out_Left) #TODO: Re-Check GCM in normal Floor heuristic: Need to use list; not dict!
    # nx.draw_circular(G, with_labels= True)
    # plt.show()
    # for i in range(N): #Debug print statements
    #     print("Node " + str(i) + " has the following edges:")
    #     for j in range(N):
    #         if i !=j and G.number_of_edges(i,j) != 0:
    #             print("To " + str(j) +" : "  + str(G.number_of_edges(i,j)))
    total_edge_cap = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i !=j:
                total_edge_cap[i,j] = rounded_matrix[i,j] + G.number_of_edges(i,j) +1
    total_edge_cap = total_edge_cap *(d /(k*N))
    # print(np.array2string(total_edge_cap)) #Debug capacity print statement
    
    thetaSH = thetaSingleHop2(total_edge_cap, modifiedM, N, input_graph=False)
    if(MH):
        return (fct.thetaEdgeFormulation(total_edge_cap,modifiedM, N, input_graph=False ), thetaSH)
    else:
        return thetaSH
def addNoisetoMatrix(M):
    #TODO: Add noise
    M = M
    #Normalize
    
    return M

#%%

if __name__ == "__main__":
    NValues=[8,16,32,48,64,128,256,512,1024]
    dE = 4
    k_s=[2,3,4,5,6]
    matrices=[]
    N= 8 
    loaded_demand = np.loadtxt(workdir+"chessboard-8" + ".mat", usecols=range(N))
    # loaded_demand = loaded_demand * dE
    eps = 1e-5
    loaded_demand[loaded_demand < eps] = 0 # Filter loaded demand?
    filtered_demand = return_normalized_matrix(loaded_demand)
    saturated_demand = filtered_demand * dE
    for k in k_s:
        print("k = " + str(k) + ": " + str(randomized_vermillion_throughput(saturated_demand, dE, k , N, MH=False)))