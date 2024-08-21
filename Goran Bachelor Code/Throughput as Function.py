import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def theta(G, M, N, d):
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


def thetaEdgeFormulation(G, M, N, d):#Given static topology G and demand matrix M, returns best throughput achievable
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

def findBestRRG(M, N, d, iter): #Given denand Matrix M, test out RRGs for given nr. of iterations and return best one with throughput and in which iter found
    best_iter = -1
    best_theta = 0
    best_G = None
    for i in range(iter):
        G_temp = nx.random_regular_graph(d,N)
        theta = thetaEdgeFormulation(G_temp, M, N, d)
        # nx.draw_circular(G_temp, with_labels= True)
        # plt.show()
        if(theta > best_theta):
            best_iter = i
            best_G = G_temp
            best_theta = theta
    return(best_iter, best_theta, best_G)

    

N=16
d=8

G = nx.random_regular_graph(d,N)


G2= nx.MultiDiGraph() #d-Strong Circle
for i in range(N):
    j = (i+1) % N
    for k in range(d): 
        keys = G2.add_edge(i,j)

G3 = nx.MultiDiGraph() #Pseudo-Chord network
for i in range(N):
    for k in range(d):
        j = (i+ 2**k)%N
        if(i == j): #Loop edges don't help us, so we add them somewhere else
            j = (j+1) % N
        key = G3.add_edge(i,j)

workdir="/home/studium/Documents/Code/rdcn-throughput/matrices/"
demand = np.loadtxt(workdir+"heatmap2.mat", usecols=range(N))
demand = demand *d
# demand = np.zeros((N,N))
# for i in range(N):
#     for j in range(N):
#         if i!=j:
#             demand[i][j] = d/(N-1)

# theta(G, demand, N, d)

# thetaEdgeFormulation(G, demand, N, d)

(iter, thetavar, BG ) = findBestRRG(demand, N, d, 10)
print(iter)
print(thetavar)


# nx.draw_circular(G, with_labels= True)
# plt.show()


# theta(G2, demand, N, d)G2= nx.MultiDiGraph() #d-Strong Circle
# G2.add_nodes_from(range(N))
# for i in range(N):
#     j = (i+1) % N
#     for k in range(d):
#         keys = G2.add_edge(i,j)
# thetaEdgeFormulation(G2, demand, N, d)
# nx.draw_circular(G2, with_labels= True)
# plt.show()
# # theta(G3, demand, N, d)
# thetaEdgeFormulation(G3, demand, N, d)
# nx.draw_circular(G3, with_labels= True)
# plt.show()

