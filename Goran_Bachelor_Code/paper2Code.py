import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import numpy as np
import Throughput_as_Function as fct
import Rounding_Draft as rd
import Floor_Draft as fd
import matplotlib.pyplot as plt
import matrixModification as MM
import random
import sys
#%%
def generate_synthmatrix_names(N):
    res = [
    "chessboard-",
    "uniform-",
    "permutation-",
    "skew-",
    "skew-",
    "skew-",
    "skew-",
    ]
    for i in range(3):
        res[i] += str(N)
    for j in range(4):
        res[j+3] += str(N) + "-0." + str(2*(j+1))
    return res
organicmatrices16 = ["data-parallelism","hybrid-parallelism","heatmap1"]
workdir="/home/vamsi/src/phd/codebase/rdcn-throughput/matrices/"
#%%
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
def randomized_vermillion_throughput(saturated_demand, saturated_noise, d, k, N, MH = True):
    normalized_demand = return_normalized_matrix(saturated_noise)
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
    
    SH_res = (thetaSingleHop2(total_edge_cap, saturated_demand, N, input_graph=False), thetaSingleHop2(total_edge_cap, saturated_noise, N, input_graph=False))
    if(MH):
        return ((fct.thetaEdgeFormulation(total_edge_cap,saturated_demand, N, input_graph=False ), fct.thetaEdgeFormulation(total_edge_cap,saturated_noise, N, input_graph=False)), SH_res)
    else:
        return SH_res


#%%

if __name__ == "__main__":
    
    NValues=[8,16,32]
    # NValues=[8]
    dE = 4
    k_s=[1,2,3,4,5,6,7,8,9,10]
    noise_values = np.linspace(0, dE/3,10)
    
    N = int(sys.argv[1])
    noise = int(sys.argv[2])

    with open("sigmetrics-throughput-results-"+str(N)+"-"+str(noise)+".csv", "w") as outputfile:
        matrices=generate_synthmatrix_names(N)
        
        if N==16:
            matrices = matrices + organicmatrices16
        
        for matrix in matrices:
            loaded_demand = np.loadtxt(workdir+matrix+".mat", usecols=range(N))
            # loaded_demand = loaded_demand * dE
            eps = 1e-5
            loaded_demand[loaded_demand < eps] = 0 # Filter loaded demand?
            filtered_demand = return_normalized_matrix(loaded_demand)
            
            saturated_noise = return_normalized_matrix(MM.add_additive_noise(filtered_demand, N, noise_values[noise])) * dE
            np.fill_diagonal(saturated_noise, 0)
            saturated_demand = filtered_demand * dE

            for k in k_s:
                res = randomized_vermillion_throughput(saturated_demand,saturated_noise ,dE, k , N)
                print(matrix, "vermConsistency", k, N, dE, noise, "add", res[0][1])
                print(matrix, "vermConsistency", k, N, dE, noise, "add", res[0][1] ,file=outputfile)
                print(matrix, "vermRobustness", k, N, dE, noise, "add", res[0][1])
                print(matrix, "vermRobustness", k, N, dE, noise, "add", res[0][1] ,file=outputfile)
                
                randC = list()
                randR = list()
                for i in range(10):
                    res = randomized_vermillion_throughput(saturated_demand,saturated_noise ,dE, random.randint(1,k+1) , N)
                    randC.append(res[0][1])
                    randR.append(res[0][0])
                
                print(matrix, "randConsistency", k, N, dE, noise, "add", np.mean(randC))
                print(matrix, "randConsistency", k, N, dE, noise, "add", np.mean(randC) ,file=outputfile)
                print(matrix, "randRobustness", k, N, dE, noise, "add", np.mean(randR))
                print(matrix, "randRobustness", k, N, dE, noise, "add", np.mean(randR) ,file=outputfile)
                
                
            saturated_noise = return_normalized_matrix(MM.add_multiplicative_noise(filtered_demand, N, noise_values[noise])) * dE
            np.fill_diagonal(saturated_noise, 0)
            saturated_demand = filtered_demand * dE

            for k in k_s:
                res = randomized_vermillion_throughput(saturated_demand,saturated_noise ,dE, k , N)
                print(matrix, "randConsistency", k, N, dE, noise, "mult", res[0][1])
                print(matrix, "randConsistency", k, N, dE, noise, "mult", res[0][1] ,file=outputfile)
                print(matrix, "randRobustness", k, N, dE, noise, "mult", res[0][0])
                print(matrix, "randRobustness", k, N, dE, noise, "mult", res[0][0] ,file=outputfile)
                
                randC = list()
                randR = list()
                for i in range(10):
                    res = randomized_vermillion_throughput(saturated_demand,saturated_noise ,dE, random.randint(1,k+1) , N)
                    randC.append(res[0][1])
                    randR.append(res[0][0])
                
                print(matrix, "randConsistency", k, N, dE, noise, "mult", np.mean(randC))
                print(matrix, "randConsistency", k, N, dE, noise, "mult", np.mean(randC) ,file=outputfile)
                print(matrix, "randRobustness", k, N, dE, noise, "mult", np.mean(randR))
                print(matrix, "randRobustness", k, N, dE, noise, "mult", np.mean(randR) ,file=outputfile)