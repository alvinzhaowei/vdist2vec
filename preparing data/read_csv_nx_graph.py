import networkx as nx
import pickle


G = nx.Graph()
count = 0
cos = {}
with open("Surat_Edgelist.csv", "r") as f:
    f.readline()
    for line in f:
        x, y, n1, n2, _, w = line.split(',')
        if n1 not in cos:
            cos[n1] = (x, y)
        G.add_weighted_edges_from([(n1, n2, float(w))])
        count += 1

# find the largest connected component
print(nx.is_connected(G))
G = G.subgraph(max(nx.connected_components(G), key=len))

node_index = {}
for i, node in enumerate(G.nodes()):
    node_index[node] = i

nG = nx.Graph()
for edge in G.edges():
    n1, n2 = edge
    w = G[n1][n2]['weight']
    n1 = node_index[n1]
    n2 = node_index[n2]
    nG.add_weighted_edges_from([(n1, n2, w)])

nx.write_weighted_edgelist(nG, "Surat_nx")


# store coordinate file
# ncos = {}
#
# for n in cos:
#     if n not in node_index:
#         continue
#     ncos[node_index[n]] = cos[n]
#
#
# pickle.dump(ncos, open("Surat_coos", "wb"))
