# Learning / Tweaking UMAP

Created in part to better understand the algorithm; tweaked to make it more pythonic; substituted the nearest-neighbour search to the version suggested in the 
LargeVis paper.

See the <a href="https://github.com/lmcinnes/">original repo</a> for multithreaded/numba code. Mine isn't actually slower for small datasets but getting to ~10000 points in 50 dimensions and I just can't match with a single threaded implementation. I do find that batches improve the convergence tremendously so I may come back to this code to parallelize/gpu-ize it. 

McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. http://arxiv.org/abs/1802.03426

Tang, J., Liu, J., Zhang, M., & Mei, Q. (2016). Visualizing Large-scale and High-dimensional Data. 287–297. https://doi.org/10.1145/2872427.2883041
