#This file contains a first rough draft of the Rounding heuristic
import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import numpy as np
import Throughput_as_Function as fct
import paperCode as pc
import matplotlib.pyplot as plt
def plotGamma(N, d, M):
    # Generate iterations list
    iterations = [1 - i * 0.01 for i in range(99)]
    gammaTheta = []
    best_theta = 0
    best_iter =100
    # Compute gammaTheta values
    for iteration in iterations:
        res = OneRoundingIter(N, d, M, iteration)

        if(res > best_theta):
            best_theta = res
            best_iter = iteration
        else:
            print("iter: ", iteration, "|res: ", res)
        gammaTheta.append(res)
    
    # Find the index and value of the maximum gammaTheta
    
    # Plot the data
    plt.figure(figsize=(8, 6))  # Optional: Set figure size
    plt.plot(iterations, gammaTheta, color='b')
    
    plt.scatter(best_iter, best_theta, color='g', s=100, zorder=5, label='Best GT')  # Highlight with a green dot
    plt.annotate(f'Best: {best_theta:.2f} In:{best_iter}', 
                 xy=(best_iter, best_theta), 
                 xytext=(best_iter - 0.1, best_theta + 0.05),
                 arrowprops=dict(facecolor='green', arrowstyle='->'),
                 fontsize=10)
    # Reverse the x-axis
    plt.gca().invert_xaxis()
    plt.ylim(0,1)
    plt.axhline(y=0.8591644204851754, color='green', linestyle='--', linewidth=1, label='Optimal Throughput')
    plt.axhline(y=0.8085981210369265, color='red', linestyle='--', linewidth=1, label='RRG Avg Throughput')
    # Add labels and title
    plt.xlabel('Iterations')
    plt.ylabel('Gamma x Theta')
    # plt.title('GammaTheta vs Iterations')
    plt.grid(True)  # Optional: Add grid lines
    plt.legend()  # Optional: Add a legend
    
    # Show the plot
    plt.show()
def OneRoundingIter(N, d, M, iteration):
    if(iteration != 1.00):
        demand = M*iteration #scale down M (demandMatrix) by factor theta (iteration)
    else: 
        demand = np.array(M)
    
    dBulk = np.floor(d*iteration).astype(int) #dIter describes how much of the edge constraint d is reserved for meeting bulk demand in rounding phase
    dRes = d - dBulk #dRes describes how much is left to construct a dRes-RRG to meet residual demand
    linkCapacity = rounding(demand, N, dBulk)
    G = nx.random_regular_graph(dRes, N)
    fct.addGraphToMatrix(G, linkCapacity)

    return(fct.thetaEdgeFormulation(linkCapacity, M, N, measure_SH=False, input_graph=False))

def rounding(M, N, d):#Given M, N and d returns rounded numpy matrix sol such that sum of all rows and all columns of sol equal to function parameter d; Assumes d-doubly stochastic matrix 
    model = gp.Model()
    entry_vars = {}
    # print(np.array2string(M))
    # print(d)
    # print("Beginning Rounding")
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
    # const = model.addVar(vtype=GRB.INTEGER, ub=0)
    diff = model.addVar(vtype=GRB.CONTINUOUS, lb=-1e10 ,ub=1e10) 
    model.addConstr(gp.quicksum(((entry_vars[i,j] - M[i,j])*entry_vars[i,j]) for i in range(N) for j in range(N) if j != i)== diff)
    #Justification for objective function, which is useful in this case?
    model.setObjective(diff, GRB.MINIMIZE) #Since this is purely a feasibility problem we optimize a constant of 0
    
    # Optimize the model
    model.optimize()
    #Create Rounded Matrix from gurobi variable values and return reference to that matrix
    sol = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i !=j:
                sol[i,j] = entry_vars[i,j].X
    # print(np.array2string(sol))
    # print("Finished Rounding")
    return sol

def thirdFinalRounding(N, d, M, measure_SH = False):
    iterations=[1-i*0.01 for i in range(99)]
    for iteration in iterations:

        demand = M*iteration #scale down M (demandMatrix) by factor theta (iteration)
        # print(np.array2string(demand))
        
        dBulk = np.floor(d*iteration).astype(int) #dIter describes how much of the edge constraint d is reserved for meeting bulk demand in rounding phase
        dRes = d - dBulk #dRes describes how much is left to construct a dRes-RRG to meet residual demand
        linkCapacity = rounding(demand, N, dBulk)

        G = nx.random_regular_graph(dRes, N)
        fct.addGraphToMatrix(G, linkCapacity)

        if(measure_SH):
            res, routed = fct.thetaEdgeFormulation(linkCapacity, demand, N, measure_SH=True, input_graph=False)
            ratioSH = routed[1] / routed[0]
            if(res ==1):
                return (iteration, ratioSH)
        else:
            res = fct.thetaEdgeFormulation(linkCapacity, demand, N, measure_SH=False, input_graph=False)
        if res== 1:
            return iteration
    return 0


if __name__ == "__main__":

    N= 16
    dE = 14
    workdir="/home/studium/Documents/Code/rdcn-throughput/matrices/"
    demandMatrix = np.loadtxt(workdir+"chessboard-16.mat", usecols=range(N))
    # fct.filtering(demandMatrix)
    # demandMatrix = fct.return_normalized_matrix(demandMatrix)
    demandMatrix = demandMatrix * dE
    # start = time.time()


    # plotGamma(8,dE, demandMatrix)
    print(plotGamma(N, dE, demandMatrix))
    # N= 8
    # dE = 7
    # workdir="/home/studium/Documents/Code/rdcn-throughput/matrices/"
    # demandMatrix = np.loadtxt(workdir+"random-skewed-8.mat", usecols=range(N))
    # # fct.filtering(demandMatrix)
    # demandMatrix = demandMatrix * dE
    # print(thirdFinalRounding(N,dE, demandMatrix))

