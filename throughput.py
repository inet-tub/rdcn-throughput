#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 13:07:08 2023

@author: vamsi
"""

import gurobipy as gp
from gurobipy import GRB
import networkx as nx
from networkx.algorithms import bipartite
import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import  random
import sys
from multiprocessing import Pool
import itertools

#%%

def createGraph(nNodes):
    G = nx.DiGraph()
    for v in range(nNodes):
        G.add_node(v)
    return G
    
def findpermutations(S,m):
    return set(itertools.permutations(S, m))


def generatePaths(G):
    paths={}

    for source in G.nodes:
        for sink in G.nodes:
            if source!=sink:
                paths[(source,sink)]=list()
                for path in nx.all_simple_edge_paths(G, source=source, target=sink,cutoff=2):
                     paths[(source,sink)].append(path)
    return paths

def modelAddFlowVars(P,m):
    flow={}
    for i in P:
        for path in P[i]:
            flow[tuple(path)]=m.addVar(lb=0,vtype="C",name='flow'+str(path))
    return flow

def modelAddTimeslotVars(schedules,m):
    timeslots={}
    for i in schedules:
        timeslots[tuple(i)]=m.addVar(lb=0, vtype="C",name='timeslot'+str(schedule))
    return timeslots

def addCapacityConstr(G,C,m,flow,paths,schedules):
    for e in G.edges:
        m.addConstr(gp.quicksum(flow[tuple(path)] for i in paths for path in paths[i] if e in path)*gp.quicksum(timeslots[tuple(schedules[i])] for i in range(len(schedules)))/gp.quicksum(c[i][(e[0],e[1])]*timeslots[tuple(schedules[i])] for i in range(len(schedules))) <= 1, name='capacity'+str(e))

#%%

nt = 4
nu = 1

switchPorts = np.arange(nt*nu)

schedules = findpermutations(switchPorts, len(switchPorts))

#%%

graphs=list()
timeslot=0
c={}

unionEdgeSet=set()

for schedule in schedules:
    G = createGraph(nt)
    c[timeslot]={}
    for i in range(nt):
        for j in range(nt):
            c[timeslot][(i,j)]=0
    for i in range(len(schedule)):
        u = int(i/nu)
        v = int(schedule[i]/nu)
        G.add_edge(u,v)
        c[timeslot][(u,v)]=1
        unionEdgeSet.add((u,v,timeslot))
    graphs.append(G)
    timeslot = timeslot+1

#%%

unionGraph = nx.MultiDiGraph()
attributesList=list()
for edge in unionEdgeSet:
    unionGraph.add_edge(edge[0],edge[1])
    # attributesList.append(edge[2])

paths = generatePaths(unionGraph)

#%%
m = gp.Model("throughput")
flows = modelAddFlowVars(paths,m)
timeslots = modelAddTimeslotVars(schedules, m)
# addCapacityConstr(unionGraph,c,m,flows,paths,schedules)