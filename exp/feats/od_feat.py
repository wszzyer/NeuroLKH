from typing import Any, List
import numpy as np
from geopandas.geodataframe import GeoDataFrame
from networkx import MultiDiGraph
from .feat import NeuralLKHFeat
from utils import smooth_matrix
from utils.lade_utils import transform_crs, SOURCE_CRS, TARGET_CRS

class ODFeat(NeuralLKHFeat):
    feat_type = "edge"

    @classmethod
    def generate_problem_meta(cls, rdf, graph: MultiDiGraph, gdf_nodes: GeoDataFrame) -> Any:
        # generate global statistics/features
        # generate OD matrix
        od = np.zeros((len(graph), len(graph)), dtype = int)
        graph_coords = gdf_nodes[["y", "x"]].values
        
        for courier_id, courier_rdf in rdf.groupby("courier_id"):
            for _, courier_day_rdf in courier_rdf.groupby(courier_rdf["delivery_time"].dt.date):
                route_coords = courier_day_rdf[["lat", "lng"]].values
                route_coords = transform_crs(route_coords, SOURCE_CRS, TARGET_CRS)
                package_to_node_dist = np.linalg.norm(route_coords[:, None] - graph_coords[None], axis=-1)
                corresponding_graph_index = package_to_node_dist.argmin(axis=-1)
                od[corresponding_graph_index[:-1], corresponding_graph_index[1:]] += 1
        # smooth OD matrix by aggregate neighbours's data.
        id_to_index = {id: index for index, id in enumerate(gdf_nodes.index)}
        adjs = [[id_to_index[v_id] for v_id in graph.neighbors(u_id)] for u_id in gdf_nodes.index] # each node's neighboures.
        return smooth_matrix(adjs, od)
    
    @classmethod
    def generate_cvrp_instance_meta(cls, problem_meta: Any, graph: MultiDiGraph, gdf_nodes: GeoDataFrame, node_indexes) -> Any:
        return problem_meta[node_indexes][:, node_indexes]
