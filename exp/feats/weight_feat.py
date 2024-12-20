from typing import Any
import numpy as np
from geopandas.geodataframe import GeoDataFrame
import networkx as nx
from .feat import RoadFeat
from joblib import parallel_config

class SSSPFeat(RoadFeat):
    feat_type = "edge"
    size = 1

    @classmethod
    def make_feat(cls, rdf, graph: nx.MultiDiGraph, gdf_nodes: GeoDataFrame) -> Any:
        coords = gdf_nodes[["x", "y"]].to_numpy()
        # Use Manhattan Distance as default
        distmat = np.abs(coords[:, np.newaxis] - coords[np.newaxis, :]).sum(-1)
        with parallel_config(n_jobs=max(len(graph) // 20, 20), verbose=0):
            shortest_path = nx.all_pairs_shortest_path_length(graph)
        mapper = dict(zip(gdf_nodes.index, np.arange(len(gdf_nodes))))
        for u, vw_dict in shortest_path:
            for v, w in vw_dict.items():
                distmat[mapper[u]][mapper[v]] = w
        cls.meta["dist"] = distmat
        return SSSPFeat((distmat + distmat.T) / 2)
    
    def __getitem__(self, node_indexes) -> Any:
        return self.data[node_indexes][:, node_indexes]

