#Notebook which provides best troughput achievable for a given M, N and d with existence of each edge being variable
#I'll probably make this into a proper python file soon, but it's more convenient this way for now
import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from gurobipy import Model, GRB
import matrixModification as mm
import Throughput_as_Function as fct
# times = []
# throughputs = []
# def log_best_throughput(model, where):
#     if where == GRB.Callback.MIPNODE or where == GRB.Callback.SIMPLEX:  # Monitor progress
#         try:
#             best_obj = model.cbGet(GRB.Callback.MIPNODE_OBJBST if where == GRB.Callback.MIPNODE else GRB.Callback.SIMPLEX_OBJBST)
#             time_passed = model.cbGet(GRB.Callback.RUNTIME)
#             if not throughputs or best_obj > throughputs[-1]:  # Only log improvements
#                 throughputs.append(best_obj)
#                 times.append(time_passed)
#         except:
#             pass  # Ignore errors if values aren't available

def perfect_theta(N, d, M):
    model = gp.Model("throughput")
    model.Params.LogToConsole = 0
    model.setParam('TimeLimit', 3600)
    #Add edge vars
    edge_vars = {}
    for i in range(N):
        for j in range(N):
            edge_vars[i, j] = model.addVar(vtype= GRB.INTEGER, name=f"edge_{i}_{j}",ub=d)
    #Inc edges constraint
    for i in range(N):
        model.addConstr(gp.quicksum(edge_vars[i,j] for j in range(N))<=d)
    #Outg edges constraint
    for i in range(N):
        model.addConstr(gp.quicksum(edge_vars[j,i] for j in range(N))<=d)

    #Add flow vars for every node combination
    flow_vars = {}
    for s in range(N):
        for d in range(N):
            if M[s,d] > 0:
                for i in range(N):
                    for j in range(N):
                        flow_vars[s, d, i, j] = model.addVar(vtype= GRB.CONTINUOUS, name=f"flow_{s}_{d}_{i}_{j}",lb=0)

    #Add throughput variable
    throughput = model.addVar(vtype=GRB.CONTINUOUS,name='throughput',lb=0,ub=1)
    
    #Source demand constraint
    for s in range(N):
        for d in range(N):
                if M[s,d] > 0:
                    model.addConstr(gp.quicksum(flow_vars[s, d, s, i] for i in range(N) if i!=s )>= M[s,d]* throughput,name =f"sdconst_{s}_{d}")

    #Dest demand constraint
    for s in range(N):
        for d in range(N):
                if M[s,d] > 0:
                    model.addConstr(gp.quicksum(flow_vars[s, d, i, d] for i in range(N) if i!=d )>= M[s,d]* throughput,name =f"ddconst_{s}_{d}")

    #Flow conservation
    for s in range(N):
        for d in range(N):
            if M[s,d] > 0:
                for j in (j for j in range(N) if j != s and j != d):
                    model.addConstr(gp.quicksum(flow_vars[s, d, i, j] for i in range(N) if i!=j )-gp.quicksum(flow_vars[s, d, j, k] for k in range(N) if k!=j)== 0,name =f"fcconst_{s}_{d}_{j}")

    #Capacity Constraints
    for i in range(N):
        for j in range(N):
            model.addConstr(gp.quicksum(flow_vars[s,d,i,j] for d in range(N) for s in range(N) if M[s,d] > 0)<= edge_vars[i,j],name =f"cconst_{s}_{d}")

    #Set the objective: maximize throughput
    model.setObjective(throughput, GRB.MAXIMIZE)

    
    model.optimize()
    # for v in model.getVars():
    #     if(v.x != 0):
    #         print(v.varName, "=", v.x)
    return(model.objVal)

    # model.optimize(log_best_throughput)
    # plt.figure(figsize=(10, 6))
    # plt.plot(times, throughputs, marker='o', linestyle='-', color='b')
    # plt.xlabel('Time (seconds)')
    # plt.ylabel('Throughput')
    # plt.title('Throughput Over Time')
    # plt.grid()
    # plt.show()
    # print(model.objVal)
if(__name__ == "__main__"):
    matrixdir="/home/studium/Documents/Code/rdcn-throughput/matrices/"
    demandMatrix = np.loadtxt(matrixdir+"topoopt"+".mat", usecols=range(16))
    fct.filtering(demandMatrix)
    demandMatrix = mm.Sinkhorn_Knopp(demandMatrix)
    demandMatrix = demandMatrix *6
    print(perfect_theta(16, 6, demandMatrix))