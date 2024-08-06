from abc import ABC, abstractmethod
from typing import List, Any
import networkx as nx
import geopandas as gpd

class NeuralLKHFeat(ABC):

    feat_type = None

    @classmethod
    @abstractmethod
    def generate_problem_meta(cls, rdf, graph:nx.MultiDiGraph, gdf_nodes: gpd.GeoDataFrame):
        raise NotImplementedError()
    
    @classmethod
    @abstractmethod
    def generate_cvrp_instance_meta(cls, problem_meta:Any, graph: nx.MultiDiGraph, gdf_nodes: gpd.GeoDataFrame, node_indexes):
        raise NotImplementedError()
    
    @classmethod
    @abstractmethod
    def generate_tsp_instance_meta(cls, problem_meta:Any, graph: nx.MultiDiGraph, gdf_nodes: gpd.GeoDataFrame, node_indexes):
        raise NotImplementedError()
    
    @classmethod
    @abstractmethod
    def generate_cvrptw_instance_meta(cls, problem_meta:Any, graph: nx.MultiDiGraph, gdf_nodes: gpd.GeoDataFrame, node_indexes):
        raise NotImplementedError()
    


