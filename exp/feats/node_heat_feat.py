from geopandas.geodataframe import GeoDataFrame
from networkx import MultiDiGraph
from .feat import NeuralLKHFeat
from utils.lade_utils import SOURCE_CRS, TARGET_CRS, transform_crs
import numpy as np

class NodeHeatFeat(NeuralLKHFeat):
    feat_type = 'node'
    size = 2
    threshold = 25
    
    @classmethod
    def generate_problem_meta(cls, rdf, graph: MultiDiGraph, gdf_nodes: GeoDataFrame):
        # TODO accept_gps_lng/accept_gps_lat 数据好像是乱的，烟台是这样
        rdf.fillna(rdf[["accept_gps_lat", "accept_gps_lng"]].mean(), inplace=True)
        accept = transform_crs(zip(rdf.accept_gps_lat, rdf.accept_gps_lng), SOURCE_CRS, TARGET_CRS)
        delivery = transform_crs(zip(rdf.lat, rdf.lng), SOURCE_CRS, TARGET_CRS)
        heat_list = []
        for index, x, y in gdf_nodes[['x', 'y']].itertuples():
            accept_heat = (((accept[:, 1] - x) ** 2 + (accept[:, 0] - y)  ** 2) <= (cls.threshold ** 2)).sum()
            delivery_heat = (((delivery[:, 1] - x) ** 2 + (delivery[:, 0] - y)  ** 2) <= (cls.threshold ** 2)).sum()
            heat_list.append((accept_heat, delivery_heat))
        return np.array(heat_list)
    
    @classmethod
    def generate_cvrp_instance_meta(cls, problem_meta, graph: MultiDiGraph, gdf_nodes: GeoDataFrame, node_indexes):
        return problem_meta[node_indexes]
