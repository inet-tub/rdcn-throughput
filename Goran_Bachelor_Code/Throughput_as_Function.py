#In this file I play around with static topologies and functions which determine the throughput they achieve
import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matrixModification as mm
import Floor_Draft as fd
import Rounding_Draft as rd
def findBestGamma(N, d, M, Rounding = False):
    # Generate iterations list
    iterations = [1 - i * 0.01 for i in range(99)]
    maxTheta = 0
    # Compute gammaTheta values
    for iteration in iterations:
        if(Rounding):
            res = rd.OneRoundingIter(N, d, M, iteration)
        else:
            res = fd.OneFloorIter(N, d, M , iteration)
        GT = res * iteration
        # print("iter: ", iteration, "|res: ", res, "|GT: ", res*iteration)
        if(GT > maxTheta):
            maxTheta = GT
        if(res == 1):
            return maxTheta

def addGraphToMatrix(G, C):
    for i, j in G.edges:
        C[i, j] += 1
        C[j, i] += 1
def match_and_increment(list_a, list_b, C):
    
    
    N = len(list_a)
    
    # Continue until all entries in list_a are zero
    while any(val > 0 for val in list_a):
        # Randomly choose an index with a value > 0 from list_a
        a_index = random.choice([i for i in range(N) if list_a[i] > 0])
        
        # Randomly choose a different index with a value > 0 from list_b

        valid_b_indices = [i for i in range(N) if list_b[i] > 0 and i != a_index]
        if not valid_b_indices:
            # print("No valid index left in list_b to decrement with")
            break
        b_index = random.choice(valid_b_indices)
        # Decrement both values at the selected indices
        list_a[a_index] -= 1
        list_b[b_index] -= 1
        
        C[a_index, b_index] +=1
        # Print for debugging purposes (optional)
    #     print(f"Decrementing A[{a_index}] and B[{b_index}]")
    #     print(f"List A: {list_a}")
    #     print(f"List B: {list_b}\n")

def createResidual(M, integer): #Modifies demand matrix M such that it becomes the residual matrix
    N = M.shape[0]
    for i in range(N):
        for j in range(N):
            if i !=j:
                M[i,j] = M[i,j]-round(integer[i,j]) #No np.max with 0, we can use negative value to determine how much capacity direct edge btwn i and j has left in rounding heuristic

import random
def filtering(M, eps=1e-5):
    # print("_________________________________________________")
    M[M < eps] = 0
    np.fill_diagonal(M, 0)
def return_normalized_matrix(M): #Normalizes a matrix by dividing it by the scalar which is the biggest row or col sum; Afterwards every row/col sum leq 1 
    max_row_sum = M.sum(axis=1).max()
    max_col_sum = M.sum(axis=0).max()
    max_sum = max(max_row_sum, max_col_sum)
    M = np.divide(M, max_sum)
    return M
def match_and_decrement(list_a, list_b, M):
    totalSH = 0
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
            # print("No valid index left in list_b to decrement with")
            break
        b_index = random.choice(valid_b_indices)
        # Decrement both values at the selected indices
        list_a[a_index] -= 1
        list_b[b_index] -= 1
        
        totalSH += min(max(M[a_index, b_index], 0),1)
        M[a_index, b_index] -=1
        # Print for debugging purposes (optional)
    #     print(f"Decrementing A[{a_index}] and B[{b_index}]")
    #     print(f"List A: {list_a}")
    #     print(f"List B: {list_b}\n")
    return totalSH
    # print("All entries in List A have been decremented to 0.")



def thetaEdgeFormulation(G, M, N, input_graph = True, measure_SH = False):#Given static topology G and demand matrix M, returns best throughput achievable
    model = gp.Model()
    capacity = {}
    model.Params.LogToConsole = 0
    total_flow =0
    SH_flow = 0
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
    # for v in model.getVars():
    #     if(v.x != 0):
    #         print(v.varName, "=", v.x)
    if(measure_SH):
        for s in range(N):
            for d in range(N):
                for i in range(N):
                    for j in range(N):
                        if i!=j and s!=d and M[s,d] != 0 and capacity[(i, j)] > 0:
                            total_flow+=flow_variables[(i, j)][(s,d)].X
                            if(i== s and d == j):
                                SH_flow+= flow_variables[(i, j)][(s,d)].X
        return (throughput.X,(total_flow, SH_flow))
    
    
    return throughput.X

def findBestRRG(M, N, d, iter, cutoff =False): #Given denand Matrix M, test out RRGs for given nr. of iterations and return best one with throughput and in which iter found
    best_iter = -1
    best_theta = 0
    best_G = None
    for i in range(iter):
        G_temp = nx.random_regular_graph(d,N)
        theta = thetaEdgeFormulation(G_temp, M, N)
        # nx.draw_circular(G_temp, with_labels= True)
        # plt.show()
        if cutoff:
            if theta < 0.8:
                return(None, 0, None)
        if(theta > best_theta):
            best_iter = i
            best_G = G_temp
            best_theta = theta
            if(theta == 1):
                return(best_iter, best_theta, best_G) # We'll never get better than 1 as throughput, so avoid calculation of next iterations
        
    return(best_iter, best_theta, best_G)

def findavgRRGtheta(M, N, d, iter):
    thetas = []
    SH = []
    for i in range(iter):
        G_temp = nx.random_regular_graph(d,N)
        theta, routed = thetaEdgeFormulation(G_temp, M, N, measure_SH=True)
        thetas.append(theta)
        SH.append(routed[1] / routed[0])
        # nx.draw_circular(G_temp, with_labels= True)
        # plt.show()
    return(np.mean(thetas), np.mean(SH))
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
    
    N=8
    dE=1
    G = createCircleGraph(N, dE)




    workdir="/home/studium/Documents/Code/rdcn-throughput/matrices/"
    demand = np.loadtxt(workdir+"skew-8-0.5.mat", usecols=range(N))
    demand2 = np.zeros((8,8))
    for i in range(8):
        for j in range(8):
            if(i == (j+1)% N):
                demand2[i,j] = 1

    print(np.array2string(demand2))
    # filtering(demand)
    # demand = mm.Sinkhorn_Knopp(demand)
    demand = demand2 *dE
    # print(findavgRRGtheta(demand, N, dE, 6))
    print(thetaEdgeFormulation(G, demand, 8, measure_SH=True))
    # demand = np.zeros((N,N))
    # for i in range(N):
    #     for j in range(N):
    #         if i!=j:
    #             demand[i][j] = d/(N-1)

    # theta(G, demand, N, d)
    # res = thetaEdgeFormulation(G, demand, N, measure_SH= True)[1]
    # print(res)
    # print(res[1] / res[0])

 
    # print(thetaEdgeFormulation(createPseudoChord(N, dE), demand, N))
    # print(thetaEdgeFormulation(createCircleGraph(N, dE), demand, N))



    # degOut = [2, 2, 2, 2, 3, 3, 3, 3, 4, 2, 3, 3, 3, 3, 3, 4]
    # degIn = [2, 3, 3, 3, 3, 3, 4, 2, 2, 2, 2, 3, 3, 3, 3, 4]
    # d = 3
    # degOut = [d] * 16
    # degIn = [d] * 16
    # graph = nx.directed_configuration_model(degIn, degOut)
    # # graph = nx.random_regular_expander_graph(N, dE, max_tries= 10000 )

    # nx.draw_circular(graph, with_labels= True)
    # plt.show()

    # graph2 = nx.random_regular_graph(d, 16)
    # nx.draw_circular(graph2, with_labels= True)
    # plt.show()
    # match_and_decrement(degOut, degIn)

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

