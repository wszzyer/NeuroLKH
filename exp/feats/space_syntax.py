import multiprocessing as mp
from itertools import cycle

import networkx as nx
import pandas as pd
import geopandas as gpd
import numpy as np

from utils.utils import map_wrapper

@map_wrapper
def calc_space_syntax(care_nodes: gpd.GeoDataFrame, radius):
    graph, gdf_nodes, gdf_edges = g_graph, g_gdf_nodes, g_gdf_edges

    care_nodes[["reachability", "integration"]] = None

    care_nodes["degree"] = [graph.degree(node) for node in care_nodes.index]
    for nodeid, row in care_nodes.iterrows():
        euc_dist = np.linalg.norm(gdf_nodes[["x", "y"]].values - row[["x", "y"]].values.astype(float), axis=-1)
        distvec = np.abs(gdf_nodes[["x", "y"]].values - row[["x", "y"]].values.astype(float)).sum(-1)
        distdict = nx.single_source_dijkstra_path_length(graph, nodeid)
        for v_i, v in enumerate(care_nodes.index):
            if v in distdict:
                distvec[v_i] = distdict[v]
        
        care_nodes.loc[nodeid, "reachability"] = distvec[euc_dist < radius].sum()
        care_nodes.loc[nodeid, "integration"] = care_nodes.loc[nodeid, "reachability"] / care_nodes.loc[nodeid, "degree"]
    return care_nodes[["reachability", "integration"]]
        

def attach_space_syntax(graph: nx.MultiDiGraph, gdf_nodes: gpd.GeoDataFrame, gdf_edges: gpd.GeoDataFrame, num_cpus=32):
    # 可达度, radius = sqrt(区域宽度)
    bounds = gdf_nodes.bounds
    # radius = np.sqrt(bounds["maxx"].max() + bounds["maxy"].max() - bounds["minx"].min() - bounds["miny"].min())
    radius = 1000 # meters

    # use global variable to avoiding process communication.
    global g_graph, g_gdf_nodes, g_gdf_edges
    g_graph, g_gdf_nodes, g_gdf_edges = graph, gdf_nodes, gdf_edges

    with mp.Pool(num_cpus) as pool:
        chunck_size = (len(gdf_nodes) + num_cpus) // num_cpus
        tasks = [gdf_nodes.iloc[i * chunck_size: (i+1) * chunck_size].copy() for i in range(num_cpus)]
        results = list(pool.imap(calc_space_syntax, zip(tasks, cycle([radius]))))
        results = pd.concat(results, axis=0)
        gdf_nodes = pd.concat([gdf_nodes, results], axis=1)
    
    return gdf_nodes