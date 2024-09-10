#In this file I play around with static topologies and functions which determine the throughput they achieve
import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt



def createResidual(M, integer): #Modifies demand matrix M such that it becomes the residual matrix
    N = M.shape[0]
    for i in range(N):
        for j in range(N):
            if i !=j:
                M[i,j] = M[i,j]-round(integer[i,j]) #No np.max with 0, we can use negative value to determine how much capacity direct edge btwn i and j has left in rounding heuristic

import random

def match_and_decrement(list_a, list_b, M):
    # Ensure both lists are of equal length
    assert len(list_a) == len(list_b), "Lists must be of equal length"
    
    # Ensure both lists have equal sums
    assert sum(list_a) == sum(list_b), "Lists must have equal sums"
    
    N = len(list_a)
    
    # Continue until all entries in list_a are zero
    while any(val > 0 for val in list_a):
        # Randomly choose an index with a value > 0 from list_a
        a_index = random.choice([i for i in range(N) if list_a[i] > 0])
        
        # Randomly choose a different index with a value > 0 from list_b

        valid_b_indices = [i for i in range(N) if list_b[i] > 0 and i != a_index]
        if not valid_b_indices:
            print("No valid index left in list_b to decrement with")
            break
        b_index = random.choice(valid_b_indices)
        # Decrement both values at the selected indices
        list_a[a_index] -= 1
        list_b[b_index] -= 1
        
        M[a_index, b_index] -=1
        # Print for debugging purposes (optional)
        print(f"Decrementing A[{a_index}] and B[{b_index}]")
        print(f"List A: {list_a}")
        print(f"List B: {list_b}\n")
    
    print("All entries in List A have been decremented to 0.")




def theta(G, M, N):#Path formulation of throughput given static topology G (currently unreliable)
    capacity = {}
    for i in range(N):
        for j in range(N):
                capacity[(i,j)] = G.number_of_edges(i,j)

    all_paths = {}
    for u in G.nodes():
        for v in G.nodes():
            if u!=v: #If is fix by Vamsi(not from slides)
                all_paths[u,v] = [list(zip(path,path[1:])) for path in nx.all_simple_paths(G, source=u,target=v)]

    model = gp.Model("throughput")

    flow_vars = {}
    for(s, d), paths in all_paths.items():
        for p in range(len(paths)):
            flow_vars[s, d, p] = model.addVar(vtype= GRB.CONTINUOUS, name=f"flow_{s}_{d}_{p}",lb=0)

    throughput = model.addVar(vtype=GRB.CONTINUOUS,name='troughput',lb=0,ub=1)

    for u,v in G.edges():
        model.addConstr(gp.quicksum(flow_vars[i,j,p] for (i,j), paths in all_paths.items() for p in range(len(paths)) if (u,v) in paths[p]) <= capacity[(u,v)],name =f"cap_{u}_{v}")

    for u in G.nodes():
        for v in G.nodes():
            if u != v:
                model.addConstr( gp.quicksum(flow_vars[u,v,p] for p in range(len(all_paths[u,v]))) >= throughput*M[u][v],name=f"M_{u}_{v}"  )

    model.setObjective(throughput, GRB.MAXIMIZE)
    model.optimize()

    # print("Throughput for the given topology and the given demand matrix is:", throughput.X)
    # for v in model.getVars():
    #     if v.x != 0:
    #             print(v.varName, "=", v.x)
    # print("________________________________________________________________")


def thetaEdgeFormulation(G, M, N, input_graph = True):#Given static topology G and demand matrix M, returns best throughput achievable
    model = gp.Model()
    capacity = {}
    model.Params.LogToConsole = 1
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
    flow_variables = {}
    for i in range(N):
        for j in range(N):
            if i != j and capacity[(i, j)] > 0:
                flow_variables[(i, j)] = {} 
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
     #Change order of loops
    for s in range(N):
        for d in range(N):
            if s != d and M[s,d] > 0:
                for j in range(N):
                    if j != s and j != d:  # Ensuring s, j, and d are all different
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

def findBestRRG(M, N, d, iter): #Given denand Matrix M, test out RRGs for given nr. of iterations and return best one with throughput and in which iter found
    best_iter = -1
    best_theta = 0
    best_G = None
    for i in range(iter):
        G_temp = nx.random_regular_graph(d,N)
        theta = thetaEdgeFormulation(G_temp, M, N)
        # nx.draw_circular(G_temp, with_labels= True)
        # plt.show()
        if(theta > best_theta):
            best_iter = i
            best_G = G_temp
            best_theta = theta
            if(theta == 1):
                return(best_iter, best_theta, best_G) # We'll never get better than 1 as throughput, so avoid calculation of next iterations
    return(best_iter, best_theta, best_G)
def createCircleGraph(N, d):
    CircleG= nx.MultiDiGraph()#d-Strong Circle
    for i in range(N):
        j = (i+1) % N
        for k in range(d): 
            keys = CircleG.add_edge(i,j)
    return CircleG
def createPseudoChord(N,d):
    ChordG = nx.MultiDiGraph() #Pseudo-Chord network
    for i in range(N):
        for k in range(d):
            j = (i+ 2**k)%N
            if(i == j): #Loop edges don't help us, so we add them somewhere else
                j = (j+1) % N
            key = ChordG.add_edge(i,j)
    return ChordG
if __name__ == "__main__":
    
    N=16
    dE=8

    G = nx.random_regular_graph(dE,N)




    workdir="/home/studium/Documents/Code/rdcn-throughput/matrices/"
    demand = np.loadtxt(workdir+"hybrid-parallelism.mat", usecols=range(N))
    demand = demand *dE
    # demand = np.zeros((N,N))
    # for i in range(N):
    #     for j in range(N):
    #         if i!=j:
    #             demand[i][j] = d/(N-1)

    # theta(G, demand, N, d)

    # thetaEdgeFormulation(G, demand, N)

 
    # print(thetaEdgeFormulation(createPseudoChord(N, dE), demand, N))
    # print(thetaEdgeFormulation(createCircleGraph(N, dE), demand, N))



    degOut = [2, 2, 2, 2, 3, 3, 3, 3, 4, 2, 3, 3, 3, 3, 3, 4]
    degIn = [2, 3, 3, 3, 3, 3, 4, 2, 2, 2, 2, 3, 3, 3, 3, 4]
    graph = nx.directed_configuration_model(degIn, degOut)
    # graph = nx.random_regular_expander_graph(N, dE, max_tries= 10000 )

    nx.draw_circular(graph, with_labels= True)
    plt.show()
    match_and_decrement(degOut, degIn)

    # theta(G2, demand, N, d)G2= nx.MultiDiGraph() #d-Strong Circle
    # G2.add_nodes_from(range(N))
    # for i in range(N):
    #     j = (i+1) % N
    #     for k in range(d):
    #         keys = G2.add_edge(i,j)
    # thetaEdgeFormulation(G2, demand, N)
    # nx.draw_circular(G2, with_labels= True)
    # plt.show()
    # # theta(G3, demand, N, d)
    # thetaEdgeFormulation(G3, demand, N)
    # nx.draw_circular(G3, with_labels= True)
    # plt.show()

