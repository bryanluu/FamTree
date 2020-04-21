#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
import pandas as pd
import igraph as ig
import cairo
import math


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
parents = G.degree(mode="in")

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
    for i in getChildren(curr):
            child = di[i]
            downlevel = levelHelper(child, node, level+1)
            if downlevel != 0:
                return downlevel
    return downlevel

def level(node, ref):
    return levelHelper(ref, node, 1)

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

# Returns greatest ancestors along paternal/maternal lines along with distance
def getGreatestAncestors(v):
    paternal = maternal = 0
    p = m = v
    while di[df.loc[p, "Father"]] != -1:
        paternal += 1
        p = di[df.loc[p, "Father"]]
    while di[df.loc[m, "Mother"]] != -1:
        maternal += 1
        m = di[df.loc[m, "Mother"]]
    return {"Paternal": (p, paternal), "Maternal": (m, maternal)}

def getGreatestAncestor(v):
    ga = getGreatestAncestors(v)
    maxline = max(ga, key=lambda k: ga[k][1])
    return ga[maxline][0]

lengths = [max(longestPath(i)) for i in range(V)]
levels = [level(v, getGreatestAncestor(v)) for v in range(V)]

layout = vs["layout"]
vs["layout"] = [[layout[k][0], levels[k]] for k in range(V)]

# plot graph
ig.plot(G, **vs)

WIDTH, HEIGHT = 256, 256

with cairo.SVGSurface("test.svg", WIDTH, HEIGHT) as surface:
    cr = cairo.Context(surface)

    cr.scale(WIDTH, HEIGHT)  # Normalizing the canvas
    cr.set_line_width(0.002)
    cr.set_source_rgb(0, 0, 0)
    cr.rectangle(0.25, 0.25, 0.5, 0.5)
    cr.stroke()

    cr.set_source_rgb(0.1, 0.1, 0.1)

    cr.select_font_face("Purisa", cairo.FONT_SLANT_NORMAL,
        cairo.FONT_WEIGHT_NORMAL)
    cr.set_font_size(13)

    cr.set_source_rgb(1, 0, 0)
    cr.set_font_size(0.25)
    cr.select_font_face("Arial",
                        cairo.FONT_SLANT_NORMAL,
                        cairo.FONT_WEIGHT_NORMAL)
    cr.move_to(0.5, 0.5)
    cr.show_text("Drawing text")

