import networkx as nx
import numpy as np

G = nx.read_weighted_edgelist("./Surat_nx")
n = len(G)
print(n)

distance_matrix = np.zeros((n, n))

# vertex index
vertex_index = {}

i = 0
for v in G.nodes():
    vertex_index[v] = i
    i += 1

for s in G.nodes():
    length = nx.single_source_dijkstra_path_length(G, s)
    s_i = vertex_index[s]
    for t in length:
        t_i = vertex_index[t]
        distance_matrix[s_i][t_i] = length[t]

print(distance_matrix)
np.save("surat_shortest_distance_matrix.npy", distance_matrix)
