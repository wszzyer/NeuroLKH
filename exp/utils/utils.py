import numpy as np
import networkx as nx

def smooth_matrix(adjs, mat: np.ndarray):
    # mat[u][v] will be aggregated by u's neighbours and v's neighbours.    
    mat_smoothed = mat.copy()
    inverse_adjs = [[] for _ in range(len(adjs))]
    for i in range(len(adjs)):
        for j in adjs[i]:
            inverse_adjs[j].append(i)
    
    for i in range(mat.shape[0]):
        mat_smoothed[i] += mat[adjs[i]].sum()
    for j in range(mat.shape[1]):
        mat_smoothed[:, j] += mat[:, inverse_adjs[j]].sum()
        
    return mat_smoothed

def calc_distmat(net, care_nodes, gdf_nodes):
    coords = gdf_nodes.loc[care_nodes][["x", "y"]].values
    # 图可能不连通，这里使用曼哈顿距离做默认值。
    distmat = np.abs(coords[:, None] - coords[None]).sum(-1)
    
    for u_i, u in enumerate(care_nodes):
        distvec = nx.single_source_dijkstra_path_length(net, u)
        for v_i, v in enumerate(care_nodes):
            if v in distvec:
                distmat[u_i][v_i] = distvec[v]
    
    return distmat
