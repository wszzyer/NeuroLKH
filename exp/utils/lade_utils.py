# maps can be visualized by https://www.qgis.org/
import os
import logging
from typing import Tuple

import osmnx
import networkx as nx
import numpy as np
import geopandas as gpd
import pandas as pd
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

maps_dir = os.path.join("./data/LaDe/road-network/")
os.makedirs(maps_dir, exist_ok=True)
name_abbr = {
    "上海市": "sh",
    "重庆市": "cq",
    "杭州市": "hz",
    "吉林市": "jl",
    "烟台市": "yt"
}
SOURCE_CRS = "EPSG:4326"
TARGET_CRS = "EPSG:32650"

def save_to_shapefile(nw_multi: nx.MultiDiGraph, map_dir_path):
    # convert undirected graph to gdfs and stringify non-numeric columns
    gdf_nodes, gdf_edges = osmnx.utils_graph.graph_to_gdfs(nw_multi)
    # gdf_edges.insert(loc=len(gdf_edges.columns), column="edgeid", value=range(len(gdf_edges)))
    gdf_nodes = osmnx.io._stringify_nonnumeric_cols(gdf_nodes)
    gdf_edges = osmnx.io._stringify_nonnumeric_cols(gdf_edges)
    # We need an unique ID for each edge
    # gdf_edges["fid"] = gdf_edges.index
    # save the nodes and edges as separate ESRI shapefiles
    os.makedirs(map_dir_path, exist_ok=True)
    gdf_nodes.to_file(os.path.join(map_dir_path, "nodes.shp"))
    gdf_edges.to_file(os.path.join(map_dir_path, "edges.shp"))

def has_map(map_name):
    exist_files = os.listdir(maps_dir)
    if any(map(lambda file: os.path.basename(file) == map_name, exist_files)):
        logger.info(f"use cached map data for {map_name}")
        return True
    else:
        logger.info(f"no cache, download from network for {map_name}")
        return False
        
def fetch_shapefile_osm_osmnx(place=None, network_type="all", map_name = None):
    if isinstance(place, str):
        nw_multi: nx.MultiDiGraph = osmnx.graph_from_place(place, network_type)
    elif isinstance(place, tuple):
        nw_multi: nx.MultiDiGraph = osmnx.graph_from_bbox(bbox=place, network_type=network_type)
    else:
        raise ValueError(f"unrecognized place {place}")
        
    edges_list = list(nw_multi.edges)
    for edge_index, edge_id in enumerate(edges_list):
        nw_multi.edges[edge_id]["edgeid"] = edge_index
    
    map_dir_path = os.path.join(maps_dir, map_name)
    save_to_shapefile(nw_multi, map_dir_path)

def get_bbox_from_coords(coords, paddings = None, percent = 2):
    """
    coords: Iterable[Tuple[latitude, longitude]]
    """
    assert percent < 50
        
    l_coords = np.percentile(coords, percent, axis=0)
    r_coords = np.percentile(coords, 100 - percent, axis=0)
    
    if paddings is None:
        paddings = np.zeros_like(l_coords)
    
    l_coords = l_coords - paddings
    r_coords = r_coords + paddings
    
    return (r_coords[0], l_coords[0], r_coords[1], l_coords[1])
    

def load_shapefile_osm_osmnx(map_name, with_graph=True, target_crs = "EPSG:32650") -> Tuple[nx.MultiDiGraph, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    target_crs: 目标参考坐标系，EPSG:32650 为投影坐标系，其坐标单位近似为（1 米）。处理中用该坐标系。
    """
    gdf_nodes = gpd.read_file(os.path.join(maps_dir, map_name, "nodes.shp")).set_index("osmid")
    gdf_edges = gpd.read_file(os.path.join(maps_dir, map_name, "edges.shp")).set_index(["u", "v", "key"])
    gdf_nodes.crs = "EPSG:4326"
    gdf_edges.crs = "EPSG:4326"
    gdf_nodes = gdf_nodes.to_crs(target_crs)
    gdf_edges = gdf_edges.to_crs(target_crs)
    gdf_edges["weight"] = gdf_edges.geometry.length
    gdf_nodes[["y", "x"]] = transform_crs(gdf_nodes[["y", "x"]].values, source_crs="EPSG:4326", target_crs=target_crs)
    graph = None
    if with_graph:
        graph = osmnx.graph_from_gdfs(gdf_nodes, gdf_edges)
    return graph, gdf_nodes, gdf_edges

def transform_crs(coords, source_crs, target_crs):
    from shapely import Point
    source_df = gpd.GeoDataFrame(geometry=[Point(y, x) for x, y in coords], crs=source_crs)
    target_df = source_df.to_crs(target_crs)
    coords2 = np.array([[p.y, p.x] for p in target_df.geometry])
    return coords2

def decode_gps_traj(traj: pd.DataFrame, lng = "lng", lat = "lat") -> None:
    """
    本段代码涉密，不要外传
    """
    traj_lat_unbiased = traj[lng].values + 2379974.967801108
    traj_lng_unbiased = traj[lat].values + 10143031.442489658
    traj_unbiased = transform_crs(np.hstack([traj_lat_unbiased, traj_lng_unbiased]), 'EPSG:3857', 'EPSG:4326')
    traj[lat] = traj_unbiased[:, 0]
    traj[lng] = traj_unbiased[:, 1]

def fetch_lade():
    if not os.path.exists("./data/LaDeArchive"):
        snapshot_download(repo_id="Cainiao-AI/LaDe", repo_type="dataset", local_dir = "./data/LaDeArchive", revision="be2cec02775cafc8d52230303f32134382bcc50b")
    