#Notebook which provides best troughput achievable for a given M, N and d with existence of each edge being variable
#I'll probably make this into a proper python file soon, but it's more convenient this way for now
import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')  
import matplotlib.pyplot as plt
from gurobipy import Model, GRB
import matrixModification as mm
import Throughput_as_Function as fct
times = []
throughputs = []
upperBounds = []
def log_theta_andUB(model, where):
    if where == GRB.Callback.MIP:  # Check for the MIP callback context
        try:
            # Retrieve runtime, upper bound, and best feasible solution
            time = model.cbGet(GRB.Callback.RUNTIME)
            upper_bound = model.cbGet(GRB.Callback.MIP_OBJBND)  # Upper bound
            best_obj = model.cbGet(GRB.Callback.MIP_OBJBST)  # Best feasible solution
            
            # Append values to lists
            times.append(time)
            throughputs.append(best_obj)
            upperBounds.append(upper_bound)
            
            # Print values for debugging
            # print(f"Time: {time}, UB: {upper_bound}, LB: {best_obj}")
        except gp.GurobiError as e:
            print(f"GurobiError: {e}")
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


def perfect_theta(N, d, M, measure_SH = False):
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

    
    # model.optimize()
    # for v in model.getVars():
    #     if(v.x != 0):
    #         print(v.varName, "=", v.x)
    
    model.optimize()

    total_flow =0
    SH_flow = 0

    if(measure_SH):
        for s in range(N):
            for d in range(N):
                for i in range(N):
                    for j in range(N):
                        if i!=j and s!=d and M[s,d] > 0:
                            total_flow+=flow_vars[s,d,i,j].X
                            if(i== s and d == j):
                                SH_flow+= flow_vars[s,d,i,j].X
        return (model.objVal,(total_flow, SH_flow))


    # model.optimize(log_theta_andUB)
    # plt.figure(figsize=(10, 6))
    # print(times)
    # print(upperBounds)
    # print(throughputs)
    # plt.plot(times, upperBounds, label='Upper Bound', color='red', linestyle='--')
    # plt.plot(times, throughputs, label='Best throughput found', color='blue', linestyle='-')



    # time_new = 15.13
    # value_new = 0.82
    # plt.scatter(time_new, value_new, label='Floor All Gamma Routed', color='yellow', s=100, zorder=3)
# #Add labels, title, and legend
    # plt.xlabel('Time (s)', fontsize = 24)
    # plt.ylabel('Objective Value', fontsize = 24)
    # # plt.title('Convergence of Upper and Lower Bounds in Optimization', fontsize = 28)
    # plt.legend(fontsize=24, loc='best')
    # plt.grid(True)
    # plt.tick_params(axis='both', which='major', labelsize=20)  # Adjust fontsize here
    # plt.ylim(0, 1)  # Objective ranges from 0 to 1
    # plt.savefig("boundconvergence2.svg", format="svg")
    # # Show the plot
    # plt.show()
    
    # plt.close()
    return(model.objVal)
if(__name__ == "__main__"):
    matrixdir="/home/studium/Documents/Code/rdcn-throughput/matrices/"
    demandMatrix = np.loadtxt(matrixdir+"skew-16-0.5"+".mat", usecols=range(16))
    # fct.filtering(demandMatrix)
    # demandMatrix = mm.Sinkhorn_Knopp(demandMatrix)
    demandMatrix = demandMatrix *14
    print(perfect_theta(16, 14, demandMatrix))