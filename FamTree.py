#!/usr/bin/env python
import numpy as np
import pandas as pd
import cairo
import argparse

# Process command-line arguments
parser = argparse.ArgumentParser(description="This program creates an SVG of family tree from CSV file.")
parser.add_argument("in_file", help="CSV input file", type=str)
parser.add_argument("out_file", help="Output filename (without extension)", type=str)
parser.add_argument("-x", "--extension", metavar="ext", help="Output file extension (SVG by default).", default=["pdf"],nargs=1, type=str)
parser.add_argument("-a", "--ancestor", metavar="root", help="Display ancestor tree of root.", nargs=1, type=str)
args = parser.parse_args()

df = pd.read_csv(args.in_file)
ext = args.extension[0]
outfile = args.out_file + "." + ext

def _getCanvas(WIDTH, HEIGHT):

    surface = None
    if ext.lower() == "svg":
        surface = cairo.SVGSurface(outfile, WIDTH, HEIGHT)
    elif ext.lower() == "pdf":
        surface = cairo.PDFSurface(outfile, WIDTH, HEIGHT)
    elif ext.lower() == "png":
        surface = cairo.ImageSurface(cairo.Format.ARGB32, WIDTH, HEIGHT)
    else:
        raise Exception("invalid extension \'." + ext + "\'")

    return surface

############## SETUP ##############

# convert to strings
df = df.astype(str)
# replace nans with blank string
df = df.applymap(lambda s: "" if s == "nan" else s)
# remove newlines
df = df.applymap(lambda s: s.rstrip('\n'))

# number of vertices
V = len(df)
# create index dictionary
di = dict()
for i in range(V):
    di[df["Full Name"][i]] = i
di[""] = -1

def getChildren(v):
    return list(map(lambda s: s.strip(), df.loc[v, "Children"].split(",")))

children = [0]*V
parents = [0]*V
spouses = {}

# fill out number of children, parents, and spouse info
for v in range(V):
    c = getChildren(v)
    children[v] = (len(c) if c != [''] else 0)
    if di[df.loc[v, "Father"]] != -1:
        parents[v] += 1
    if di[df.loc[v, "Father"]] != -1:
        parents[v] += 1
    spouses[v] = int(di[df.loc[v, "Spouse"]])

def height(node, down=True):
    if down:
        if children[node] == 0:
            return -1
        else:
            return max(height(e[1]) for e in E if e[0] == node) + 1
    else:
        if parents[node] == 0:
            return -1
        else:
            pa, ma = di[df.loc[node, "Father"]], di[df.loc[node, "Mother"]]
            return max(height(pa, down), height(ma, down)) + 1

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

if(args.ancestor is None):
    print("Family members:", V)

    ######################################## FULL TREE ########################################

    TREE_WIDTH = max(tw*100, (np.max(years) - np.min(years))*10)
    TREE_HEIGHT = TREE_WIDTH
    WIDTH, HEIGHT = TREE_WIDTH, TREE_HEIGHT
    NEIGHBOR_SPACING = TREE_WIDTH/tw
    BORDER_SCALE = 0.1
    FONT_SIZE = 0.015

    surface = _getCanvas(WIDTH, HEIGHT)

    # Labels
    labels = [df.loc[v, "Nickname"] if len(df.loc[v, "Nickname"]) > 0 else df.loc[v, "Full Name"] for v in range(V)]

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

    # Branch heights
    top_branch = {-1:-1}
    low_branch = {-1:-1}
    for v in range(V):
        top_branch[v] = max(vy[v], vy[spouses[v]]) + 1/(My-my)
    for v in range(V):
        low_branch[v] = min([vy[di[c]] for c in getChildren(v)]) - 5/(My-my)

    # Node colors
    colors = {-1:np.array([0,0,0])}
    for v in range(V):
        if parents[i[v]] == 0:
            colors[i[v]] = np.random.random(3)
        else:
            pa = di[df.loc[i[v], "Father"]]
            ma = di[df.loc[i[v], "Mother"]]
            colors[i[v]] = (colors[pa]+colors[ma])/2

    cr = cairo.Context(surface)

    cr.scale(WIDTH, HEIGHT)  # Normalizing the canvas

    for v in range(V):
        # if node has children, draw tree paths
        if children[v] > 0:
            # Draw joining line
            cr.set_source_rgb(*((colors[v]+colors[spouses[v]])/2))
            cr.move_to(vx[v], vy[v])
            cr.set_line_width(0.001)
            cr.line_to(vx[v], top_branch[v])
            cr.stroke()
            cr.move_to(vx[v], top_branch[v])
            cr.line_to((vx[v]+vx[spouses[v]])/2, top_branch[v])
            cr.stroke()

            # Draw line to children
            cr.move_to((vx[v]+vx[spouses[v]])/2, top_branch[v])
            cr.line_to((vx[v]+vx[spouses[v]])/2, low_branch[v])
            cr.stroke()

        if parents[v] > 0:
            # draw pre-connector branch
            cr.set_source_rgb(*colors[v])
            pa, ma = di[df.loc[v, "Father"]], di[df.loc[v, "Mother"]]
            cl = max(low_branch[pa], low_branch[ma])
            cr.move_to((vx[pa]+vx[ma])/2, cl)
            cr.line_to(vx[v], cl)
            cr.stroke()

            # draw node connection
            cr.set_line_width(0.001)
            cr.move_to(vx[v], cl)
            cr.line_to(vx[v], vy[v])
            cr.stroke()

        # Draw node
        cr.arc(vx[v], vy[v], vr, 0, 2*np.pi)
        cr.set_source_rgb(*colors[v])
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

    print("Family Tree written to", outfile)

######################################## ANCESTRY TREE ########################################
else:
    # print(args.ancestor)
    # Root variable
    ancroot = args.ancestor[0]
    root = -1
    try:
        root = di[ancroot]
    except KeyError:
        raise Exception("Invalid family member name \'" + ancroot + "\'.")

    # Ancestry Tree of root
    VERTICAL_SPACING = 100
    th = height(root, down=False)+1
    tw = 2**th
    md = max(tw, th)
    TREE_WIDTH = md*VERTICAL_SPACING
    TREE_HEIGHT = md*VERTICAL_SPACING
    WIDTH, HEIGHT = TREE_WIDTH, TREE_HEIGHT
    NEIGHBOR_SPACING = TREE_WIDTH/tw
    BORDER_SCALE = 0.1
    FONT_SIZE = 0.015

    surface = _getCanvas(WIDTH, HEIGHT)

    # Labels
    labels = [df.loc[v, "Nickname"] if len(df.loc[v, "Nickname"]) > 0 else df.loc[v, "Full Name"] for v in range(V)]

    # Vertex positions
    vx = np.zeros(V)
    vy = np.zeros(V)

    # Set x position based on relative width compared to neighbors
    def _ancestorSetXPos(v, lvl=1):
        if children[v] == 0:
            vx[v] = WIDTH/2

        pa, ma = di[df.loc[v, "Father"]], di[df.loc[v, "Mother"]]
        if(pa != -1):
            if(parents[v] == 1):
                vx[pa] = vx[v]
            else:
                vx[pa] = vx[v]-NEIGHBOR_SPACING*lvl
            _ancestorSetXPos(pa, lvl/2)
        if(ma != -1):
            if(parents[v] == 1):
                vx[ma] = vx[v]
            else:
                vx[ma] = vx[v]+NEIGHBOR_SPACING*lvl
            _ancestorSetXPos(ma, lvl/2)

    # Set y position based on height:
    def _ancestorSetYPos(v):
        if children[v] == 0:
            vy[v] = TREE_HEIGHT

        pa, ma = di[df.loc[v, "Father"]], di[df.loc[v, "Mother"]]
        if(pa != -1):
            vy[pa] = vy[v] - TREE_HEIGHT/th
            _ancestorSetYPos(pa)
        if(ma != -1):
            vy[ma] = vy[v] - TREE_HEIGHT/th
            _ancestorSetYPos(ma)

    _ancestorSetXPos(root)
    _ancestorSetYPos(root)

    # Normalize positions
    mx, Mx = np.min(vx), np.max(vx)
    my, My = np.min(vy), np.max(vy)
    vx = (BORDER_SCALE*WIDTH + (vx - mx)/(Mx - mx)*WIDTH)/((1+2*BORDER_SCALE)*WIDTH)
    vy = (BORDER_SCALE*HEIGHT + (vy - my)/(My - my)*HEIGHT)/((1+2*BORDER_SCALE)*HEIGHT)

    # Vertex size
    vr = 0.01

    # Node colors
    colors = {-1:np.array([0,0,0])}
    for v in range(V):
        if parents[i[v]] == 0:
            colors[i[v]] = np.random.random(3)
        else:
            pa = di[df.loc[i[v], "Father"]]
            ma = di[df.loc[i[v], "Mother"]]
            colors[i[v]] = (colors[pa]+colors[ma])/2

    cr = cairo.Context(surface)

    cr.scale(WIDTH, HEIGHT)  # Normalizing the canvas

    q = []

    q.append(root)

    while(len(q) > 0):
        v = q.pop(0)

        # Draw node
        cr.arc(vx[v], vy[v], vr, 0, 2*np.pi)
        cr.set_source_rgb(*colors[v])
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


#         print("Nodes:", labels[v])

        pa, ma = di[df.loc[v, "Father"]], di[df.loc[v, "Mother"]]

        # Draw branches
        if(parents[v] > 0):
            cr.set_source_rgb(*colors[v])
            cr.move_to(vx[v], vy[v])
            cr.set_line_width(0.001)
            cr.line_to(vx[v], (vy[v] + max(vy[pa], vy[ma]))/2)
            cr.stroke()
            cr.move_to(vx[v], (vy[v] + max(vy[pa], vy[ma]))/2)
            cr.line_to(vx[pa], (vy[v] + max(vy[pa], vy[ma]))/2)
            cr.stroke()
            cr.move_to(vx[pa], (vy[v] + max(vy[pa], vy[ma]))/2)
            cr.line_to(vx[pa], vy[pa])
            cr.stroke()
            cr.move_to(vx[v], (vy[v] + max(vy[pa], vy[ma]))/2)
            cr.line_to(vx[ma], (vy[v] + max(vy[pa], vy[ma]))/2)
            cr.stroke()
            cr.move_to(vx[ma], (vy[v] + max(vy[pa], vy[ma]))/2)
            cr.line_to(vx[ma], vy[ma])
            cr.stroke()
            if pa != -1:
                q.append(pa)
            if ma != -1:
                q.append(ma)

    print("Ancestry Tree of " + args.ancestor[0] + " written to", outfile)

if ext.lower() == "png":
    surface.write_to_png(outfile)  # Output to PNG

surface.finish()
