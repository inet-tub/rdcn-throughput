import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def thetaEdgeFormulation(G, M, N):#Given static topology G and demand matrix M, returns best throughput achievable
    model = gp.Model()
    capacity = {}
    # model.Params.LogToConsole = 0
    # model.Params.OptimalityTol = 1e-4

    for i in range(N):
        for j in range(N):
                capacity[(i,j)] = G.number_of_edges(i,j)
    flow_variables = {}
    for i in range(N):
        for j in range(N):
            if i != j and capacity[(i, j)] > 0:
                flow_variables[(i, j)] = {} #TODO: Remove flow variables where there is no edge, in general remove all variables that are unused
                #Also if there is no demand between a pair we also don't need flow variables
                for s in range(N):
                    for d in range(N):
                        if s != d and M[s,d] > 0:
                            # Create a flow variable for each node pair (i, j) corresponding to node pairs (s, d)
                            flow_variables[(i, j)][(s, d)] = model.addVar(vtype=GRB.CONTINUOUS, name=f'flow_{i}_{j}_{s}_{d}', lb=0)
    throughput = model.addVar(vtype=GRB.CONTINUOUS, name='throughput', lb=0, ub=1)
    model.update()
                # Implement the source demand constraints for all s in N and all d in N
    for s in range(N):
        for d in range(N):
            if s != d and M[s,d] > 0:
                # Define the source demand constraint
                source_demand_constraint_expr = gp.quicksum(flow_variables[(s, i)][(s, d)] for i in range(N) if i != s and capacity[(s,i)] > 0) >= throughput * M[s, d]
                model.addConstr(source_demand_constraint_expr, f'source_demand_constraint_{s}_{d}')
    
    # Implement the destination demand constraints for all s in N and all d in N
    for s in range(N):
        for d in range(N):
            if s != d and M[s,d] > 0:
                # Define the destination demand constraint
                dest_demand_constraint_expr = gp.quicksum(flow_variables[(i, d)][(s, d)] for i in range(N) if i != d and capacity[(i,d)] > 0) >= throughput * M[s, d]
                model.addConstr(dest_demand_constraint_expr, f'dest_demand_constraint_{s}_{d}')
    
    # Implement the flow conservation constraints for all j in N (excluding s and d), s in N, and d in N
    for j in range(N):
        for s in range(N):
            for d in range(N):
                if j != s and j != d and s != d and M[s,d] > 0 :  # Ensuring s, j, and d are all different
                    # Define the flow conservation constraint
                    flow_conservation_constraint_expr = gp.quicksum(flow_variables[(i, j)][(s, d)] for i in range(N) if i != j and capacity[(i, j)] > 0) - gp.quicksum(flow_variables[(j, k)][(s, d)] for k in range(N) if k != j and capacity[(j, k)] > 0) == 0
                    model.addConstr(flow_conservation_constraint_expr, f'flow_conservation_constraint_{j}_{s}_{d}')
    
    # Implement the capacity constraints for every i in N, for every j in N
    for i in range(N):
        for j in range(N):
            if i != j and capacity[(i,j)] >0:
                capacity_constraint_expr = gp.quicksum(flow_variables[(i, j)][(s, d)] for s in range(N) for d in range(N) if s != d and M[s,d] > 0) <= capacity[(i, j)]
                model.addConstr(capacity_constraint_expr, f'capacity_constraint_{i}_{j}')
    
    # Set the objective to maximize throughput
    model.setObjective(throughput, GRB.MAXIMIZE)
    
    # Optimize the model
    model.optimize()
    return throughput.X