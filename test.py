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
# ig.plot(G, **vs)

AGE_GAP = 10

# sort birth years
years = np.array([int(df.loc[v, "Year Born"]) for v in range(V)])
i = np.argsort(years)

# max tree width
tw = 0

# width at the given vertex's age group
W = np.ones(V, int)

# age group
k = 0
K = np.zeros(V, int) # age group array

# find age group and neighbors for each vertex
v = 0
while v < V-1:
#     print(years[i][v])

    vstart = v

    while(v+1 < V):
#         print(W[i])
        k = vstart
        # check if next birth is within age gap, add to width if it is
        if years[i][v+1] - years[i][v] < AGE_GAP:
            W[i[vstart:v+2]] = W[i[vstart]]
            K[i[vstart:v+2]] = k
            W[i[vstart:v+2]] += 1
            if W[i[vstart]] > tw:
                tw = W[i[vstart]]
            v += 1
        else:
            v += 1
            break

print("Neighbors:", W[i])
print("Age Groups:", K[i])

print("Tree Width:", tw)

TREE_WIDTH = max(tw*100, (np.max(years) - np.min(years))*10)
TREE_HEIGHT = TREE_WIDTH
WIDTH, HEIGHT = TREE_WIDTH, TREE_HEIGHT
NEIGHBOR_SPACING = TREE_WIDTH/tw
BORDER_SCALE = 0.1

# Vertex positions
vx = np.zeros(V)
vy = np.array([years[v] for v in range(V)])

# Get x position based on relative width compared to neighbors
X = np.arange(V)
for v in range(V):
    g = (K == K[v])
    if v > np.min(X[g]):
        continue
    else:
        for n in range(W[v]):
            vx[X[g][n]] = TREE_WIDTH/2 - NEIGHBOR_SPACING*W[v]/2 + n*NEIGHBOR_SPACING

# Normalize positions
mx, Mx = np.min(vx), np.max(vx)
my, My = np.min(vy), np.max(vy)
vx = (BORDER_SCALE*WIDTH + (vx - mx)/(Mx - mx)*WIDTH)/((1+2*BORDER_SCALE)*WIDTH)
vy = (BORDER_SCALE*HEIGHT + (vy - my)/(My - my)*HEIGHT)/((1+2*BORDER_SCALE)*HEIGHT)

# Vertex size
vr = 0.005

FONT_SIZE = 0.015
labels = [df.loc[v, "Nickname"] if len(df.loc[v, "Nickname"]) > 0 else df.loc[v, "Full Name"] for v in range(V)]


with cairo.SVGSurface("test.svg", WIDTH, HEIGHT) as surface:
    cr = cairo.Context(surface)

    cr.scale(WIDTH, HEIGHT)  # Normalizing the canvas

    # Draw vertices
    for v in range(V):
        # Draw node
        cr.arc(vx[v], vy[v], vr, 0, 2*math.pi)
        cr.set_source_rgb(0.8, 0, 0)
        cr.fill()

        # Draw label
        cr.set_source_rgb(0, 0, 0)
        cr.select_font_face("Arial",
                cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        cr.set_font_size(0.015)
        _xbearing, _ybearing, _width, _height, _xadvance, _yadvance = (
                    cr.text_extents(labels[v]))
        cr.move_to(vx[v] - _xbearing - _width / 2,
                    vy[v] - 0.02 - _ybearing - _height / 2)
        cr.show_text(labels[v])
