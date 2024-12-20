import multiprocessing as mp
from itertools import cycle

import networkx as nx
import pandas as pd
import geopandas as gpd
from geopandas.geodataframe import GeoDataFrame
from networkx import MultiDiGraph
import numpy as np

from utils.utils import map_wrapper
from .feat import RoadFeat


class SpaceSyntaxFeat(RoadFeat):
    feat_type = 'node'
    size = 2
    
    @classmethod
    def make_feat(cls, rdf, graph: MultiDiGraph, gdf_nodes: GeoDataFrame):
        # radius = np.sqrt(bounds["maxx"].max() + bounds["maxy"].max() - bounds["minx"].min() - bounds["miny"].min())
        radius = 1000 # meters
        if "dist" not in cls.meta:
            raise RuntimeError("One edge weight must be calculated before using space syntax feat.")
        dist_mat = cls.meta["dist"]
        # FIXME: Know what they are
        reachability = np.ma.masked_array(dist_mat, dist_mat < radius).sum(axis=-1)
        integration = reachability / np.array(graph.degree())[:, 1]
        return SpaceSyntaxFeat(np.stack((reachability, integration), axis=-1))
    
    
    def __getitem__(self, node_indexes):
        return self.data[node_indexes]
