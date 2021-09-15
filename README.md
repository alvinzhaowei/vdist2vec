# vdist2vec
This is the implementation of vdist2vec model in the following paper: \
Jianzhong Qi, Wei Wang, Rui Zhang, Zhuowei Zhao∗. "[A Learning Based Approach to Predict Shortest-Path Distances](https://openproceedings.org/2020/conf/edbt/paper_215.pdf)", EDBT 2020

# Problem definition

We consider a road network graph <img height="10" src=./equations/eq1.gif>, where V is a set of n vertices  (road intersections)  and E is a set of m edges (roads). 
A vertex <img height="10" src=./equations/eq2.gif> has a pair of geo-coordinates. An edge <img height="10" src=./equations/eq3.gif> connects two vertices <img height="10" src=./equations/vi.gif> and <img height="10" src=./equations/vj.gif>, and has a weight <img height="10" src=./equations/eq4.gif>, which represents the  distance to travel across the edge. For simplicity, in what follows, our discussions assume undirected edges, although our techniques also work for directed edges. 

A path <img height="10" src=./equations/eq5.gif> between vertices 
<img height="10" src=./equations/vi.gif> and <img height="10" src=./equations/vj.gif> consists of a sequence of vertices <img height="10" src=./equations/eq6.gif> 
such that there is an edge between any two adjacent vertices in the sequence.  The length of <img height="10" src=./equations/eq5.gif>, denoted by <img height="10" src=./equations/eq7.gif>, is the sum of the weights of the edges between  adjacent vertices in <img height="10" src=./equations/eq5.gif>, i.e., 
<img height="10" src=./equations/eq8.gif>
We are interested in the path  <img height="10" src=./equations/eq9.gif> between <img height="10" src=./equations/vi.gif> and <img height="10" src=./equations/vj.gif> with the smallest length, i.e., the shortest path. 
Its length is the (shortest-path) distance <img height="10" src=./equations/eq10.gif> between <img height="10" src=./equations/vi.gif> and <img height="10" src=./equations/vj.gif>, i.e., 
<img height="10" src=./equations/eq11.gif>.
We aim to predict <img height="10" src=./equations/eq10.gif> given <img height="10" src=./equations/vi.gif> and <img height="10" src=./equations/vj.gif> with a high accuracy and efficiency, which is defined as the shortest-path distance query. 

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
