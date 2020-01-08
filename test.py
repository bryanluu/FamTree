#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
import pandas as pd
import igraph as ig


# In[78]:


df = pd.read_csv("data/test.csv")

# convert to strings
df = df.astype(str)
# replace nans with blank string
df = df.applymap(lambda s: "" if s == "nan" else s)
# remove newlines
df = df.applymap(lambda s: s.rstrip('\n'))

# display data
df


# In[128]:


# number of vertices
V = len(df)
# create index dictionary
di = dict()
for i in range(V):
    di[df["Full Name"][i]] = i
di[""] = -1

def getChildren(v):
    return list(map(lambda s: s.strip(), df.loc[v, "Children"].split(",")))

# create edges
E = [(source, di[dest]) for source in range(V) \
        for dest in list(map(lambda s: s.strip(), df.loc[source, "Children"].split(","))) \
        if dest != ""]

# create graph
G = ig.Graph(directed = True)
G.add_vertices(V)
G.add_edges(E)

# visual styles
vs = {}
vs["vertex_label"] = [df.loc[v, "Nickname"] if df.loc[v, "Nickname"] != "" \
                        else df.loc[v, "Full Name"] for v in range(V)]
vs["layout"] = G.layout("rt")

children = G.degree(mode="out")

def topologicalSortUtil(v, visited, stack):
    if v == -1:
        return

    # mark node v as visited
    visited[v] = True

    # recurse for adjacent nodes
    for name in getChildren(v):
        child = di[name]
        if not visited[child]:
            topologicalSortUtil(child, visited, stack)

    # push vertex to stack to store result
    stack.append(v)

def topologicalSort():
    visited = [False]*V
    stack = []

    for i in range(V):
        if not visited[i]:
            topologicalSortUtil(i, visited, stack)

    return stack

def longestPath(v):
    topsort = topologicalSort()
    dist = [-1]*V
    dist[v] = 0

    while(len(topsort) != 0):
        u = topsort.pop()
        if dist[u] != -1:
            for name in getChildren(u):
                child = di[name]
                if dist[child] < dist[u] + 1:
                    dist[child] = dist[u] + 1
    return dist

lengths = [max(longestPath(i)) for i in range(V)]

layout = vs["layout"]
vs["layout"] = [[layout[k][0], -lengths[k]] for k in range(V)]

# plot graph
ig.plot(G, **vs)

