from geopandas.geodataframe import GeoDataFrame
from networkx import MultiDiGraph
from .feat import RoadFeat
from utils.lade_utils import SOURCE_CRS, TARGET_CRS, transform_crs
import numpy as np

class NodeHeatFeat(RoadFeat):
    feat_type = 'node'
    size = 2
    threshold = 25
    
    @classmethod
    def make_feat(cls, rdf, graph: MultiDiGraph, gdf_nodes: GeoDataFrame):
        # TODO accept_gps_lng/accept_gps_lat 数据好像是乱的，烟台是这样
        rdf.fillna(rdf[["accept_gps_lat", "accept_gps_lng"]].mean(), inplace=True)
        accept = transform_crs(zip(rdf.accept_gps_lat, rdf.accept_gps_lng), SOURCE_CRS, TARGET_CRS)
        delivery = transform_crs(zip(rdf.lat, rdf.lng), SOURCE_CRS, TARGET_CRS)
        heat_list = []
        for index, x, y in gdf_nodes[['x', 'y']].itertuples():
            accept_heat = (((accept[:, 1] - x) ** 2 + (accept[:, 0] - y)  ** 2) <= (cls.threshold ** 2)).sum()
            delivery_heat = (((delivery[:, 1] - x) ** 2 + (delivery[:, 0] - y)  ** 2) <= (cls.threshold ** 2)).sum()
            heat_list.append((accept_heat, delivery_heat))
        return NodeHeatFeat(np.array(heat_list))
    
    def __getitem__(self, node_indexes):
        return self.data[node_indexes]
