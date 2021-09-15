# vdist2vec
vdist2vec model in paper **A Learning Based Approach to Predict Shortest-Path Distances**

## Problem definition
We consider a road network graph ```G = \langle V, E \rangle```, where ```V``` is a set of $n$ vertices  (road intersections)  and $E$ is a set of $m$ edges (roads). 
A vertex $v_i \in V$ has a pair of geo-coordinates. An edge $e_{i,j} \in E$  connects two vertices $v_i$ and $v_j$, and has 
 a \emph{weight}  $e_{i,j}.w$, which represents the  distance to travel across the edge. 
 Fig.~\ref{fig:motivation}a shows an example, where $v_1, v_2, ... , v_5$ are the vertices, and the numbers on the edges are the weights. 
 For simplicity, 
in what follows, our discussions assume undirected edges, 
although our techniques also work for directed edges. 

A path $p_{i,j}$ between vertices 
$v_i$ and $v_j$ consists of a sequence of vertices $v_i \rightarrow v_1 \rightarrow  v_2 \rightarrow  ... \rightarrow v_x \rightarrow v_j$ 
such that there is an edge between any two adjacent vertices in the sequence.  The \emph{length} of $p_{i,j}$, denoted by $|p_{i,j}|$, is the sum of the weights of the edges between  adjacent vertices in $p_{i,j}$, i.e., 
$|p_{i, j}| = e_{i, 1}.w + e_{1, 2}.w + ... + e_{x, j}.w.$
We are interested in the path  $p_{i, j}^*$ between $v_i$ and $v_j$ with the smallest length, i.e., the \emph{shortest path}. 
Its length is the \emph{(shortest-path) distance} $d(v_i, v_j)$ between $v_i$ and $v_j$, i.e., 
$d(v_i, v_j) = |p_{i, j}^*|$.
Consider vertices $v_1$ and $v_5$ in Fig.~\ref{fig:motivation}a. 
Their distance $d(v_1, v_5) = 4$ is the length of path $v_1 \rightarrow v_4 \rightarrow v_5$. 
We aim to predict $d(v_i, v_j)$ given $v_i$ and $v_j$ with a high  accuracy and efficiency, which is defined as the \emph{shortest-path distance query}. 
