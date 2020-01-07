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
vs["vertex_label"] = df["Full Name"]
vs["layout"] = G.layout("rt")

children = G.degree(mode="out")

def height(node):
    if children[node] == 0:
        return -1
    else:
        return max(height(e[1]) for e in E if e[0] == node) + 1

def levelHelper(curr, node, level):
    if curr == -1:
        return 0

    if curr == node:
        return level

    downlevel = 0
    for e in E:
        if e[0] == curr:
            child = e[1]
            downlevel = levelHelper(child, node, level+1)
            if downlevel != 0:
                return downlevel
    return downlevel

def level(node):
    return levelHelper(di["Paternal Grandfather"], node, 1)

layout = vs["layout"]
vs["layout"] = [[3*layout[k][0], 3*level(k)] for k in range(V)]

# plot graph
ig.plot(G, **vs)

