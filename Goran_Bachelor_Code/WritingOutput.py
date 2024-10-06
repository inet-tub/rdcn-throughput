import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import numpy as np
import random_Matrices as rM
import Throughput_as_Function as fct
import Floor_Draft as fd
import Rounding_Draft as rd
import EVariable as ev
import sys

def generate_synthmatrix_names(N):
    res = [
    "chessboard-",
    "uniform-",
    "permutation-",
    "random-skewed-",
    "skew-",
    "skew-",
    "skew-",
    "skew-",
    "skew-",
    "skew-",
    "skew-",
    "skew-",
    "skew-",
    ]
    for i in range(4):
        res[i] += str(N)
    for j in range(9):
        res[j+4] += str(N) + "-0." + str(j+1)
    return res
organicmatrices16 = ["data-parallelism","hybrid-parallelism","heatmap1","heatmap2","heatmap3", "topoopt"]
matrixdir="/home/studium/Documents/Code/rdcn-throughput/matrices/"
outputdir = "/home/studium/Documents/Code/rdcn-throughput/Goran_Bachelor_Code/"
n_values_used = [8,16]
d_s = [2,3,4,5,6,7]

def writePerfectTheta():
    outputfile = open(outputdir+"output3", "w")
    outputfile.write("N matrix d Alg NA throughput\n")
    print("N matrix d Alg NA throughput")
    for N in n_values_used:
        matrices = generate_synthmatrix_names(N)
        if(N == 16):
            matrices+=organicmatrices16
        for matrix in matrices:
            demandMatrix = np.loadtxt(matrixdir+matrix+".mat", usecols=range(N))
            fct.filtering(demandMatrix)
            demandMatrix = fct.return_normalized_matrix(demandMatrix)
            for d in d_s:
                d = round(d*(N/8))
                saturatedM = demandMatrix * d
                string_Beginning = str(N) +" "+ matrix +" " + str(d)

                perfect_theta= ev.perfect_theta(N,d,saturatedM)
                perfect_string = string_Beginning + " perfect " + "NA " + str(perfect_theta)
                print(perfect_string)
                outputfile.write(perfect_string+"\n")
if __name__ == "__main__":

    RRG_Iter = 6
    outputfile = open(outputdir+"outputTest", "w")
    outputfile.write("N matrix d Alg RRGIter throughput\n")
    print("N matrix d Alg RRGIter throughput")
    

    for N in n_values_used:
        matrices = generate_synthmatrix_names(N)
        if(N == 16):
            matrices+=organicmatrices16
        for matrix in matrices:
            demandMatrix = np.loadtxt(matrixdir+matrix+".mat", usecols=range(N))
            fct.filtering(demandMatrix)
            demandMatrix = fct.return_normalized_matrix(demandMatrix)
            for d in d_s:
                d = round(d*(N/8))
                saturatedM = demandMatrix * d


                


                string_Beginning = str(N) +" "+ matrix +" " + str(d)


                (_,RRG_theta,_) =  fct.findBestRRG(saturatedM, N, d, RRG_Iter)
                RRG_string = string_Beginning + " RRG "+ str(RRG_Iter) + " " +str(RRG_theta)
                print(RRG_string)
                outputfile.write(RRG_string+"\n")

                Floor_theta = fd.thetaByFloor(N,d,saturatedM, RRG_Iter)
                Floor_String = string_Beginning + " Floor " + str(RRG_Iter) +" " + str(Floor_theta)
                print(Floor_String)
                outputfile.write( Floor_String+"\n")

                Rounding_theta = rd.alternativeTheta(N, d, saturatedM, RRG_Iter)
                Rounding_String =string_Beginning +  " Rounding " + str(RRG_Iter) +" " + str(Rounding_theta)
                print(Rounding_String)
                outputfile.write( Rounding_String+"\n")

                circle_theta = fct.thetaEdgeFormulation(fct.createCircleGraph(N,d),saturatedM, N)
                circleString = string_Beginning + " Circle " +"NA" + " " +str(circle_theta)
                print(circleString)
                outputfile.write( circleString+"\n")

                chord_theta = fct.thetaEdgeFormulation(fct.createPseudoChord(N,d), saturatedM, N)
                chordString = string_Beginning + " Chord " +"NA" + " " +str(chord_theta)
                print(chordString)
                outputfile.write( chordString+"\n")
    writePerfectTheta()








            
    # # floorResults =  []
    # # roundingResults = []
    # randomRes = []
    # expanderRes =[]
    # CircleRes = []
    # ChordRes = []
    # ExpanderRes = []


 
    # print(np.array2string(matrix))
    # fct.preprocessing(matrix)
    # print(np.array2string(matrix))
    # matrix = matrix * dE
    # print(fd.thetaByFloor(N, dE, matrix, 5))
    # print(matrices16)
    # for matrix in matrices16:
    #     demandMatrix = np.loadtxt(workdir+matrix+".mat", usecols=range(N)) #TODO: Manually remove tiny entries
    #     # print(matrix + "__________________________________")
    #     # print(np.array2string(demandMatrix))
    #     # print("After filtering:___________________________________________")

    #     fct.filtering(demandMatrix)
    #     # print(np.array2string(demandMatrix))
    #     # print("After normalizing:_____________________________________________")
    #     demandMatrix = fct.return_normalized_matrix(demandMatrix)
    #     # print(np.array2string(demandMatrix))
    #     demand = demandMatrix *dE
    #     # ExpanderRes.append(fct.thetaEdgeFormulation(nx.random_regular_expander_graph(N, dE, max_tries= 10000 ),demand, N))
    #     # floorResults.append(fd.thetaByFloor(N,dE, demand, 6))
    #     roundingResults.append(rd.alternativeTheta(N,dE, demand, 6)) #Alternative Rounding!
    #     # CircleRes.append(fct.thetaEdgeFormulation(fct.createCircleGraph(N, dE), demand, N))
    #     # ChordRes.append(fct.thetaEdgeFormulation(fct.createPseudoChord(N,dE), demand, N))

    # # print("Results for each matrix using Floor Method:")
    # # for i in range(len(matrices16)):
    # #     print(matrices16[i] + ": " + str(floorResults[i]))
    # print("Results for each matrix using Rounding Method:")
    # for j in range(len(matrices16)):
    #     print(matrices16[j] + ": " + str(roundingResults[j]))
