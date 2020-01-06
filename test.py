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

# plot graph
ig.plot(G, **vs)

