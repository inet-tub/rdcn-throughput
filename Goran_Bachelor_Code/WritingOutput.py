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
import matrixModification as mm

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
def oneIteration(N, M, matrix, RRG_Iter, outputfile):
    if(matrix in organicmatrices16):
        M = mm.Sinkhorn_Knopp(M)
        matrix = "Sinkhorn_" + matrix
    for d in d_s:
        d = round(d*(N/8))
        saturatedM = M * d


        string_Beginning = str(N) +" "+ matrix +" " + str(d)


        # RRG_theta =  fct.findavgRRGtheta(saturatedM, N, d, RRG_Iter)
        # RRG_string = string_Beginning + " RRG "+ str(RRG_Iter) + " " +str(RRG_theta)
        # print(RRG_string)
        # outputfile.write(RRG_string+"\n")

        # Floor_theta = fd.thetaByFloor(N,d,saturatedM, RRG_Iter)
        # Floor_String = string_Beginning + " Floor " + "NA" +" " + str(Floor_theta)
        # print(Floor_String)
        # outputfile.write( Floor_String+"\n")

        old_Rounding_theta = rd.thetaByRounding(N, d, saturatedM, RRG_Iter)
        old_Rounding_String =string_Beginning +  " Old_Rounding " + "NA" +" " + str(old_Rounding_theta)
        print(old_Rounding_String)
        outputfile.write(old_Rounding_String+"\n")


        Rounding_theta = rd.altThetawGCM(N, d, saturatedM, RRG_Iter)
        Rounding_String =string_Beginning +  " Rounding " + "NA" +" " + str(Rounding_theta)
        print(Rounding_String)
        outputfile.write( Rounding_String+"\n")


        # alt_Rounding_theta = rd.alternativeTheta(N, d, saturatedM, RRG_Iter)
        # Rounding_String =string_Beginning +  " Alt_Rounding " + "NA" +" " + str(alt_Rounding_theta)
        # print(Rounding_String)
        # outputfile.write( Rounding_String+"\n")

        # circle_theta = fct.thetaEdgeFormulation(fct.createCircleGraph(N,d),saturatedM, N)
        # circleString = string_Beginning + " Circle " +"NA" + " " +str(circle_theta)
        # print(circleString)
        # outputfile.write( circleString+"\n")

        # chord_theta = fct.thetaEdgeFormulation(fct.createPseudoChord(N,d), saturatedM, N)
        # chordString = string_Beginning + " Chord " +"NA" + " " +str(chord_theta)
        # print(chordString)
        # outputfile.write( chordString+"\n")

def writePerfectTheta(N, matrices, d_lists):
    for i in range(len(matrices)):
        matrix = matrices[i]
        demandMatrix = np.loadtxt(matrixdir+matrix+".mat", usecols=range(N))
        fct.filtering(demandMatrix)
        demandMatrix = fct.return_normalized_matrix(demandMatrix)
        demandMatrix = mm.Sinkhorn_Knopp(demandMatrix) #Currently IN
        for d in d_lists[i]:
            saturatedM = demandMatrix * d
            string_Beginning = str(N) +" "+matrix +" " + str(d)

            perfect_theta= ev.perfect_theta(N,d,saturatedM)
            perfect_string = string_Beginning + " Optimal " + "NA " + str(perfect_theta)
            print(perfect_string)

if __name__ == "__main__":
    # all_ds = [4,6,8,10,12,14]
    ds = [16,2]
    matrices = organicmatrices16 +generate_synthmatrix_names(16)
    d_lists = [ds,ds,ds,ds,ds,ds,ds,ds,ds,ds,ds,ds,ds,ds,ds,ds,ds,ds,ds,ds,ds,ds,ds]

    # writePerfectTheta(16, matrices, d_lists)
    # RRG_Iter = 10
    # print("N matrix d Alg RRGIter throughput")
    # outputfile = open(outputdir+"RoundingCompare", "w")
    # outputfile.write("N matrix d Alg RRGIter throughput\n")
    writePerfectTheta(16,matrices, d_lists)
    # for N in n_values_used:
    #     matrices = generate_synthmatrix_names(N)
    #     if(N == 16):
    #         matrices = organicmatrices16 + matrices
    #     for matrix in matrices:
    #         demandMatrix = np.loadtxt(matrixdir+matrix+".mat", usecols=range(N))
    #         fct.filtering(demandMatrix)
    #         demandMatrix = fct.return_normalized_matrix(demandMatrix)
    #         oneIteration(N, demandMatrix, matrix, RRG_Iter, outputfile)
    #         if(matrix in organicmatrices16):
    #             demandMatrix = mm.Sinkhorn_Knopp(demandMatrix)
    #             matrix = "Sinkhorn_" + matrix
    #             oneIteration(N, demandMatrix, matrix, RRG_Iter, outputfile)