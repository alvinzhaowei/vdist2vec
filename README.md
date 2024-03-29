# vdist2vec
This is the implementation of vdist2vec model in the following paper: \
Jianzhong Qi, Wei Wang, Rui Zhang, Zhuowei Zhao. "[A Learning Based Approach to Predict Shortest-Path Distances](https://openproceedings.org/2020/conf/edbt/paper_215.pdf)", EDBT 2020

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

### Preparing the input data
We use Python package “NetworkX” (https://networkx.org/) to handle road network. NetworkX has many build-in functions such as connected components algorithm, and shortest-path distance algorithm. It is a handy tool to play around with graphs. Would be better to check the version installed and refer to the corresponding documentation as their APIs could vary a lot from version to version. Please note that, no mater what kind of data type we used for vertex in NetworkX, after saved to file and reloaded, the data type of vertex become string.



You may read the CSV file (or other format) and create a NetworkX graph. Some road networks downloaded from website are CSV files, such as (https://figshare.com/articles/dataset/Urban_Road_Network_Data/2061897). One CSV file (Surat_Edgelist.csv) and a sample code (read_csv_nx_graph.py) to read it into NetworkX graph can be found in the preparing data folder.

A code sample (test.py) that shows how to read a NetworkX graph from a file and calculate the shortest path distance matrix (input of vdist2vec) is inlcuded, too.



## Results
<p align="center">
  <img src=./figure/results.PNG>
</p>

***Baselines***:

***landmark-bt***:

Frank W. Takes and Walter A. Kosters. 2014. Adaptive landmark selection
strategies for fast shortest path computation in large real-world graphs. In
WI-IAT.

***landmark-km***: 

it uses the k vertices that are the closest to the vertex kmeans centroids (computed in Euclidean space) as the landmarks;

***ado***:

Jagan Sankaranarayanan and Hanan Samet. 2009. Distance oracles for spatial
networks. In ICDE.

***geodnn***: 

Ishan Jindal, Xuewen Chen, Matthew Nokleby, Jieping Ye, et al. 2017. A unified
neural network approach for estimating travel time and distance for a taxi
trip. arXiv preprint arXiv:1710.04350 (2017).

***node2vec***: 

Fatemeh Salehi Rizi, Joerg Schloetterer, and Michael Granitzer. 2018. Shortest
path distance approximation using deep learning techniques. In ASONAM.

## Citation
If you find this repository useful in your research, please cite the following paper:

```
@inproceedings{qi2020learning,
  title={A learning based approach to predict shortest-path distances},
  author={Qi, Jianzhong and Wang, Wei and Zhang, Rui and Zhao, Zhuowei},
  booktitle={23rd International Conference on Extending Database Technology (EDBT)},
  pages={367--370},
  year={2020}
}
```
