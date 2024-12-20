from abc import ABC, abstractmethod
from typing import Any, Dict
import networkx as nx
import geopandas as gpd

class RoadFeat(ABC):

    feat_type = None
    meta = {}

    def __init__(self, data: Any):
        self.data = data

    @classmethod
    @abstractmethod
    def make_feat(cls, rdf, meta: Dict[str, Any], graph: nx.MultiDiGraph, gdf_nodes: gpd.GeoDataFrame):
        raise NotImplementedError()
