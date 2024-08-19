from typing import Any, List
import numpy as np
from geopandas.geodataframe import GeoDataFrame
from networkx import MultiDiGraph
from .feat import NeuralLKHFeat
from utils import calc_distmat

class SSSPFeat(NeuralLKHFeat):
    feat_type = "edge"
    size = 1

    @classmethod
    def generate_problem_meta(cls, rdf, graph: MultiDiGraph, gdf_nodes: GeoDataFrame) -> Any:
        return None
    
    @classmethod
    def _generate_weight(cls, graph, gdf_nodes, node_indexes):
        distmat = calc_distmat(graph, gdf_nodes.index[node_indexes], gdf_nodes)
        distmat = (distmat + distmat.T) / 2
        return distmat.astype(int)

    @classmethod
    def generate_cvrp_instance_meta(cls, problem_meta: Any, graph: MultiDiGraph, gdf_nodes: GeoDataFrame, node_indexes) -> Any:
        return cls._generate_weight(graph, gdf_nodes, node_indexes)
    
    @classmethod
    def generate_tsp_instance_meta(cls, problem_meta: Any, graph: MultiDiGraph, gdf_nodes: GeoDataFrame, node_indexes) -> Any:
        return cls._generate_weight(graph, gdf_nodes, node_indexes)
    
    @classmethod
    def generate_cvrptw_instance_meta(cls, problem_meta: Any, graph: MultiDiGraph, gdf_nodes: GeoDataFrame, node_indexes) -> Any:
        return cls._generate_weight(graph, gdf_nodes, node_indexes)
