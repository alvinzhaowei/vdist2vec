# vdist2vec
This is the implementation of vdist2vec model in the following paper: \
Jianzhong Qi, Wei Wang, Rui Zhang, Zhuowei Zhao∗. "[A Learning Based Approach to Predict Shortest-Path Distances](https://openproceedings.org/2020/conf/edbt/paper_215.pdf)", EDBT 2020

# Problem definition

We consider a road network graph <img src=./equations/eq1.gif>, where V is a set of n vertices  (road intersections)  and E is a set of m edges (roads). 
A vertex <img src=./equations/eq2.gif> has a pair of geo-coordinates. An edge <img src=./equations/eq3.gif> connects two vertices <img src=./equations/vi.gif> and <img src=./equations/vj.gif>, and has a weight <img src=./equations/eq4.gif>, which represents the  distance to travel across the edge. For simplicity, in what follows, our discussions assume undirected edges, although our techniques also work for directed edges. 

A path <img src=./equations/eq5.gif> between vertices 
<img src=./equations/vi.gif> and <img src=./equations/vj.gif> consists of a sequence of vertices <img src=./equations/eq6.gif> 
such that there is an edge between any two adjacent vertices in the sequence.  The length of <img src=./equations/eq5.gif>, denoted by <img src=./equations/eq7.gif>, is the sum of the weights of the edges between  adjacent vertices in <img src=./equations/eq5.gif>, i.e., 
<img src=./equations/eq8.gif>
We are interested in the path  <img src=./equations/eq9.gif> between <img height="10" src=./equations/vi.gif> and <img src=./equations/vj.gif> with the smallest length, i.e., the shortest path. 
Its length is the (shortest-path) distance <img src=./equations/eq10.gif> between <img src=./equations/vi.gif> and <img src=./equations/vj.gif>, i.e., 
<img src=./equations/eq11.gif>.
We aim to predict <img src=./equations/eq10.gif> given <img src=./equations/vi.gif> and <img src=./equations/vj.gif> with a high accuracy and efficiency, which is defined as the shortest-path distance query. 
# Model
<p align="center">
  <img src=./figure/model.PNG>
</p>

## Data
The datasets are from:

Camil Demetrescu, Andrew Goldberg, and David Johnson. 2006. 9th DIMACS implementation challenge–Shortest Paths. American Math. Society (2006).

Alireza Karduni, Amirhassan Kermanshah, and Sybil Derrible. 2016. A protocol to convert spatial polyline data to network formats and applications to world urban road networks. Scientific Data 3 (2016), 160046.

## Results
<p align="center">
  <img src=./figure/results.PNG>
</p>

*Baselines*:

*landmark-bt*:

it uses the top-k vertices passed by the largest numbers of shortest paths between the vertex pairs as the landmarks; 

Frank W. Takes and Walter A. Kosters. 2014. Adaptive landmark selection
strategies for fast shortest path computation in large real-world graphs. In
WI-IAT.

*landmark-km*: 

it uses the k vertices that are the closest to the vertex kmeans centroids (computed in Euclidean space) as the landmarks;

*ado*:

it recursively partitions the vertices into subsets of well separated vertices and stores the distance between subsets to
approximate the distance between vertices (we tune its approximation parameter ϵ such that it has a similar space cost to ours);

Jagan Sankaranarayanan and Hanan Samet. 2009. Distance oracles for spatial
networks. In ICDE.

*geodnn*: 

it trains an MLP to predict the distance of two vertices given their geo-coordinates (we use its recommended settings); 

Ishan Jindal, Xuewen Chen, Matthew Nokleby, Jieping Ye, et al. 2017. A unified
neural network approach for estimating travel time and distance for a taxi
trip. arXiv preprint arXiv:1710.04350 (2017).

*node2vec*: 

it uses node2vec to learn vertex embeddings and trains an MLP to predict vertex distances given the learned embeddings (we use its recommended settings).

Fatemeh Salehi Rizi, Joerg Schloetterer, and Michael Granitzer. 2018. Shortest
path distance approximation using deep learning techniques. In ASONAM.

## Citation
If you find this repository useful in your research, please cite the following paper:

```
@article{qi2020learning,
  title={A learning based approach to predict shortest-path distances},
  author={Qi, Jianzhong and Wang, Wei and Zhang, Rui and Zhao, Zhuowei},
  year={2020},
  publisher={Open Proceedings}
}
```
