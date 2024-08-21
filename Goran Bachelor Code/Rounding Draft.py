import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import numpy as np

def thetaEdgeFormulation(G, M, N):#Given static topology G and demand matrix M, returns best throughput achievable
    model = gp.Model()
    capacity = {}
    for i in range(N):
        for j in range(N):
                capacity[(i,j)] = G.number_of_edges(i,j)
    flow_variables = {}
    for i in range(N):
        for j in range(N):
            if i != j:
                flow_variables[(i, j)] = {}
                for s in range(N):
                    for d in range(N):
                        if s != d:
                            # Create a flow variable for each node pair (i, j) corresponding to node pairs (s, d)
                            flow_variables[(i, j)][(s, d)] = model.addVar(vtype=GRB.CONTINUOUS, name=f'flow_{i}_{j}_{s}_{d}', lb=0)
    throughput = model.addVar(vtype=GRB.CONTINUOUS, name='throughput', lb=0, ub=1)
    model.update()
                # Implement the source demand constraints for all s in N and all d in N
    for s in range(N):
        for d in range(N):
            if s != d:
                # Define the source demand constraint
                source_demand_constraint_expr = gp.quicksum(flow_variables[(s, i)][(s, d)] for i in range(N) if i != s) >= throughput * M[s, d]
                model.addConstr(source_demand_constraint_expr, f'source_demand_constraint_{s}_{d}')
    
    # Implement the destination demand constraints for all s in N and all d in N
    for s in range(N):
        for d in range(N):
            if s != d:
                # Define the destination demand constraint
                dest_demand_constraint_expr = gp.quicksum(flow_variables[(i, d)][(s, d)] for i in range(N) if i != d) >= throughput * M[s, d]
                model.addConstr(dest_demand_constraint_expr, f'dest_demand_constraint_{s}_{d}')
    
    # Implement the flow conservation constraints for all j in N (excluding s and d), s in N, and d in N
    for j in range(N):
        for s in range(N):
            for d in range(N):
                if j != s and j != d and s != d:  # Ensuring s, j, and d are all different
                    # Define the flow conservation constraint
                    flow_conservation_constraint_expr = gp.quicksum(flow_variables[(i, j)][(s, d)] for i in range(N) if i != j) - gp.quicksum(flow_variables[(j, k)][(s, d)] for k in range(N) if k != j) == 0
                    model.addConstr(flow_conservation_constraint_expr, f'flow_conservation_constraint_{j}_{s}_{d}')
    
    # Implement the capacity constraints for every i in N, for every j in N
    for i in range(N):
        for j in range(N):
            if i != j:
                capacity_constraint_expr = gp.quicksum(flow_variables[(i, j)][(s, d)] for s in range(N) for d in range(N) if s != d) <= capacity[(i, j)]
                model.addConstr(capacity_constraint_expr, f'capacity_constraint_{i}_{j}')
    
    # Set the objective to maximize throughput
    model.setObjective(throughput, GRB.MAXIMIZE)
    
    # Optimize the model
    model.optimize()
    # print("_______________________________________________________")
    return throughput.X

# def findBestRRG(M, N, d, iter): #Given denand Matrix M, test out RRGs for given nr. of iterations and return best one with throughput and in which iter found
#     best_iter = -1
#     best_theta = 0
#     best_G = None
#     for i in range(iter):
#         G_temp = nx.random_regular_graph(d,N)
#         theta = thetaEdgeFormulation(G_temp, M, N, d)
#         # nx.draw_circular(G_temp, with_labels= True)
#         # plt.show()
#         if(theta > best_theta):
#             best_iter = i
#             best_G = G_temp
#             best_theta = theta
#     return(best_iter, best_theta, best_G)

def rounding(M, N, d):
    model = gp.Model()
    entry_vars = {}
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
    model.setObjective(const, GRB.MAXIMIZE) 
    
    # Optimize the model
    model.optimize()
    #Create Rounded Matrix and return reference to that matrix
    sol = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i !=j:
                sol[i,j] = entry_vars[i,j].X
    return sol
def createResidual(M, rounded):
    for i in range(N):
        for j in range(N):
            if i !=j:
                M[i,j] = np.max((M[i,j]-rounded[i,j], 0))

N= 16
dE = 8
workdir="/home/studium/Documents/Code/rdcn-throughput/matrices/"
demandMatrix = np.loadtxt(workdir+"heatmap2.mat", usecols=range(N))
demandMatrix = demandMatrix * dE
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


iterations=[1-i*0.01 for i in range(99)]
finalIteration = 0.1

for iteration in iterations:
    print("################")
    print(iteration)
    print("################")
    demand = demandMatrix*iteration
    print(np.array2string(demand))
    print("___________________________-")
    dIter = np.floor(dE*iteration).astype(int)
    intMatrix = rounding(demand, N, dIter)
    createResidual(demand, intMatrix)
    print(np.array2string(demand))
    print(dE - dIter)
    G = nx.random_regular_graph(dE - dIter, N)
    if thetaEdgeFormulation(G, demand, N) == 1:
        finalIteration = iteration
        print(finalIteration)
        break
    print(np.array2string(demand))
    print(dIter)
    print()
    print("________________________")
