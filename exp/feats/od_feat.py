from typing import Any, List
import numpy as np
from geopandas.geodataframe import GeoDataFrame
from networkx import MultiDiGraph
from .feat import RoadFeat
from utils import smooth_matrix
from utils.lade_utils import transform_crs, SOURCE_CRS, TARGET_CRS

class ODFeat(RoadFeat):
    feat_type = "edge"
    size = 1

    @classmethod
    def make_feat(cls, rdf, graph: MultiDiGraph, gdf_nodes: GeoDataFrame) -> Any:
        # generate global statistics/features
        # generate OD matrix
        od = np.zeros((len(graph), len(graph)), dtype = int)
        
        for courier_id, courier_rdf in rdf.groupby("courier_id"):
            for _, courier_day_rdf in courier_rdf.groupby(courier_rdf["delivery_time"].dt.date):
                od[courier_day_rdf.graph_index[:-1], courier_day_rdf.graph_index[1:]] += 1
        # smooth OD matrix by aggregate neighbours's data.
        id_to_index = {id: index for index, id in enumerate(gdf_nodes.index)}
        adjs = [[id_to_index[v_id] for v_id in graph.neighbors(u_id)] for u_id in gdf_nodes.index] # each node's neighboures.
        return ODFeat(smooth_matrix(adjs, od))
    
    def __getitem__(self, node_indexes) -> Any:
        return self.data[node_indexes][:, node_indexes]
