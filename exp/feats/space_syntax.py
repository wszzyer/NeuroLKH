import multiprocessing as mp
from itertools import cycle

import networkx as nx
import pandas as pd
import geopandas as gpd
from geopandas.geodataframe import GeoDataFrame
from networkx import MultiDiGraph
import numpy as np

from utils.utils import map_wrapper
from .feat import NeuralLKHFeat


class SpaceSyntaxFeat(NeuralLKHFeat):
    feat_type = 'node'
    size = 2
    
    @classmethod
    def generate_problem_meta(cls, rdf, graph: MultiDiGraph, gdf_nodes: GeoDataFrame):
        # radius = np.sqrt(bounds["maxx"].max() + bounds["maxy"].max() - bounds["minx"].min() - bounds["miny"].min())
        radius = 1000 # meters

        # use global variable to avoiding process communication.
        global g_graph, g_gdf_nodes
        g_graph, g_gdf_nodes = graph, gdf_nodes

        num_cpus = 16
        with mp.Pool(num_cpus) as pool:
            chunck_size = (len(gdf_nodes) + num_cpus - 1) // num_cpus
            tasks = [gdf_nodes.iloc[i * chunck_size: (i+1) * chunck_size].copy() for i in range(num_cpus)]
            results = list(pool.imap(calc_space_syntax, zip(tasks, cycle([radius]))))
            ss_df = pd.concat(results, axis=0)
        return ss_df.values.astype(float)
    
    @classmethod
    def generate_cvrp_instance_meta(cls, problem_meta, graph: MultiDiGraph, gdf_nodes: GeoDataFrame, node_indexes):
        return problem_meta[node_indexes]


@map_wrapper
def calc_space_syntax(care_nodes: gpd.GeoDataFrame, radius):
    graph, gdf_nodes = g_graph, g_gdf_nodes

    care_nodes[["reachability", "integration"]] = None

    care_nodes["degree"] = [graph.degree(node) for node in care_nodes.index]
    for nodeid, row in care_nodes.iterrows():
        euc_dist = np.linalg.norm(gdf_nodes[["x", "y"]].values - row[["x", "y"]].values.astype(float), axis=-1)
        # 使用曼哈顿距离作为默认值，以应对图不连通
        distvec = np.abs(gdf_nodes[["x", "y"]].values - row[["x", "y"]].values.astype(float)).sum(-1)
        distdict = nx.single_source_dijkstra_path_length(graph, nodeid)
        for v_i, v in enumerate(care_nodes.index):
            if v in distdict:
                distvec[v_i] = distdict[v]
        
        care_nodes.loc[nodeid, "reachability"] = distvec[euc_dist < radius].sum()
        care_nodes.loc[nodeid, "integration"] = care_nodes.loc[nodeid, "reachability"] / care_nodes.loc[nodeid, "degree"]
    return care_nodes[["reachability", "integration"]]