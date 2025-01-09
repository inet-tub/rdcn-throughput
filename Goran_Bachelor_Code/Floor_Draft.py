#This file contains a first rough draft of the Rounding heuristic
import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import numpy as np
import Throughput_as_Function as fct
import matplotlib.pyplot as plt
import time

def plotGamma(N, d, M):
    # Generate iterations list
    iterations = [1 - i * 0.01 for i in range(99)]
    gammaTheta = []
    best_theta = 0
    best_iter =100
    # Compute gammaTheta values
    for iteration in iterations:
        res = OneFloorIter(N, d, M, iteration)
        if(res > best_theta):
            best_theta = res
            best_iter = iteration
        else:
            print("iter: ", iteration, "|res: ", res)
        gammaTheta.append(res)
    
    
    # Plot the data
    plt.figure(figsize=(8, 6))  # Optional: Set figure size
    plt.plot(iterations, gammaTheta, color='b')
    
    # Highlight the maximum value
    # plt.scatter(ret_iter, max_gammaTheta, color='r', s=100, zorder=5, label='Return Gamma')  # Highlight with a red dot
    # plt.annotate(f'Return: {max_gammaTheta:.2f}', 
    #              xy=(max_iteration, max_gammaTheta), 
    #              xytext=(max_iteration - 0.1, max_gammaTheta + 0.05),
    #              arrowprops=dict(facecolor='red', arrowstyle='->'),
    #              fontsize=10)
    plt.scatter(best_iter, best_theta, color='g', s=100, zorder=5)  # Highlight with a green dot
    plt.annotate(f'θ*: {best_theta:.3f} With γ:{best_iter:.2f}', 
                 xy=(best_iter, best_theta), 
                 xytext=(best_iter - 0.1, best_theta + 0.05),
                 arrowprops=dict(facecolor='green', arrowstyle='->'),
                 fontsize=10)
    # Reverse the x-axis
    plt.gca().invert_xaxis()
    
    plt.axhline(y=0.8363289560560487, color='green', linestyle='--', linewidth=1, label='Optimal Throughput')
    plt.axhline(y=0.44475453191253356, color='red', linestyle='--', linewidth=1, label='RRG Avg Throughput')
    # Add labels and title
    plt.xlabel('Gamma', fontsize=20)
    plt.ylabel('Throughput achieved',fontsize=20)
    plt.ylim(0,1)
    # plt.title('GammaTheta vs Iterations')
    plt.grid(True)  # Optional: Add grid lines
    plt.legend()  # Optional: Add a legend
    plt.savefig(f"FloorThetaEvo.svg", format="svg", dpi=300)
    # Show the plot
    plt.show()
def OneFloorIter(N, d, M, iteration):#Returns throughput that can be achieved for given N, d and M with floor heuristic. RRGiter determines how many RRGs you will try per iter
    # print(M)
    if(iteration != 1.00):
        demand = M*iteration #scale down M (demandMatrix) by factor theta (iteration)
    else: 
        demand = np.array(M)
    # np.savetxt('demandBefore.txt', demand)
    # print(np.array2string(demand,formatter={'float_kind':lambda x: "%.5f" % x}))
    linkCapacity = np.floor(demand)
    # print(linkCapacity)
    # np.savetxt('demandFloor.txt', linkCapacity)


    outDegree = [0]*N # How many outgoing links each node has left
    inDegree = [0]*N # How many incoming links each node has left
    for row in range(N):
        outDegree[row] = int(np.min([d,N-1]) - np.sum(linkCapacity[row]))
    for column in range(N):
        inDegree[column] = int(np.min([d,N-1]) - np.sum(linkCapacity[:,column]))

    # print(outDegree)
    # print(inDegree)
    dRes = np.min([N-1, np.min([outDegree,inDegree])])#Smallest val 


    outLeft = [ x - dRes for x in outDegree]
    inLeft = [ x - dRes for x in inDegree]
    # print(outLeft)
    # print(inLeft)
    
    G = nx.random_regular_graph(dRes,N)
    fct.addGraphToMatrix(G, linkCapacity)
    fct.match_and_increment(outLeft, inLeft, linkCapacity)
    # print(linkCapacity)
    res = fct.thetaEdgeFormulation(linkCapacity, M, N, input_graph=False)
    return res


def ThetaByFloor(N, d, M, measure_SH = False):#Returns throughput that can be achieved for given N, d and M with floor heuristic. RRGiter determines how many RRGs you will try per iter
    iterations=[1-i*0.01 for i in range(99)]
    for iteration in iterations:


    
        if(iteration != 1.00):
            demand = M*iteration #scale down M (demandMatrix) by factor theta (iteration)
        else: 
            demand = np.array(M)
        linkCapacity = np.floor(demand) #FloorMatrix

        outDegree = [0]*N # How many outgoing links each node has left
        inDegree = [0]*N # How many incoming links each node has left
        for row in range(N):
            outDegree[row] = int(np.min([d,N-1]) - np.sum(linkCapacity[row]))
        for column in range(N):
            inDegree[column] = int(np.min([d,N-1]) - np.sum(linkCapacity[:,column]))
        dRes = np.min([N-1, np.min([outDegree,inDegree])])#Smallest val 
        outLeft = [ x - dRes for x in outDegree]
        inLeft = [ x - dRes for x in inDegree]

        G = nx.random_regular_graph(dRes,N)
        fct.addGraphToMatrix(G, linkCapacity)
        fct.match_and_increment(outLeft, inLeft, linkCapacity)
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
    dE = 6
    workdir="/home/studium/Documents/Code/rdcn-throughput/matrices/"
    demandMatrix = np.loadtxt(workdir+"skew-16-0.9.mat", usecols=range(N))
    # fct.filtering(demandMatrix)
    # demandMatrix = fct.return_normalized_matrix(demandMatrix)
    demandMatrix = demandMatrix * dE
    # start = time.time()


    # plotGamma(8,dE, demandMatrix)
    print(plotGamma(N, dE, demandMatrix))

    # print(OneFloorIter(8, dE, demandMatrix, 0.7))

    # end = time.time()
    # length = end - start
    # print(length)
    # plotGamma(N, dE, demandMatrix)

