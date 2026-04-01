import streamlit as st
import pandas as pd
import torch
import numpy as np
import os
import copy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import h3.api.basic_str as h3_api
import contextily as cx
from shapely.geometry import Polygon, Point, LineString
import geopandas as gpd

from mutilEnv import HexTrafficEnv  
from mutilDqfsAgent import ExpertDQN 

# --- Base Config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_FOLDER, exist_ok=True)

ANCHOR_LAT, ANCHOR_LON = 42.995486, -81.253178
H3_RES = 9
ANCHOR_CELL = h3_api.latlng_to_cell(ANCHOR_LAT, ANCHOR_LON, H3_RES)

# --- H3 Utils ---
def get_h3_polygon(i, j):
    try:
        anchor_ij = h3_api.cell_to_local_ij(ANCHOR_CELL, ANCHOR_CELL)
        target_cell = h3_api.local_ij_to_cell(
            ANCHOR_CELL, i + anchor_ij[0], j + anchor_ij[1]
        )
        boundary = h3_api.cell_to_boundary(target_cell)
        return Polygon([(lon, lat) for lat, lon in boundary])
    except:
        return None

def get_latlon_point(i, j):
    try:
        anchor_ij = h3_api.cell_to_local_ij(ANCHOR_CELL, ANCHOR_CELL)
        target_cell = h3_api.local_ij_to_cell(
            ANCHOR_CELL, i + anchor_ij[0], j + anchor_ij[1]
        )
        lat, lon = h3_api.cell_to_latlng(target_cell)
        return lon, lat
    except:
        return None

# --- Draw Map ---
def draw_replay_map(env, tracks, model_name, ep_idx):
    fig, ax = plt.subplots(figsize=(8, 8))

    # roads
    road_polys = [get_h3_polygon(i, j) for i, j in env.london_main_roads]
    gdf_road = gpd.GeoDataFrame(
        geometry=[p for p in road_polys if p], crs="EPSG:4326"
    )
    if not gdf_road.empty:
        gdf_road.to_crs(epsg=3857).plot(
            ax=ax, facecolor='none', edgecolor='blue', linewidth=0.1, alpha=0.05
        )

    # chargers
    l2 = [get_h3_polygon(i, j) for (i, j), lv in env.charging_stations.items() if lv == 'L2']
    l3 = [get_h3_polygon(i, j) for (i, j), lv in env.charging_stations.items() if lv == 'L3']

    if l2:
        gpd.GeoDataFrame(geometry=[p.centroid for p in l2], crs="EPSG:4326")\
            .to_crs(epsg=3857).plot(ax=ax, color='cyan', marker='+', markersize=50, label='L2 Charger')

    if l3:
        gpd.GeoDataFrame(geometry=[p.centroid for p in l3], crs="EPSG:4326")\
            .to_crs(epsg=3857).plot(ax=ax, color='red', marker='P', markersize=80, label='L3 Fast Charger')

    # agents
    for i, track in enumerate(tracks):
        pts = []
        for p in track:
            pt = get_latlon_point(p[0], p[1])
            if pt:
                pts.append(pt)

        if len(pts) < 2:
            continue

        soc = env.vehicles[i].soc if i < len(env.vehicles) else 0.5
        color = mcolors.to_hex((1 - soc, soc, 0))  # ✅ FIX

        # path
        gpd.GeoDataFrame(geometry=[LineString(pts)], crs="EPSG:4326")\
            .to_crs(epsg=3857).plot(ax=ax, color=color, linewidth=2)

        # start
        gpd.GeoDataFrame(geometry=[Point(pts[0])], crs="EPSG:4326")\
            .to_crs(epsg=3857).plot(ax=ax, color=color, marker='o', markersize=30)

        # end
        gpd.GeoDataFrame(geometry=[Point(pts[-1])], crs="EPSG:4326")\
            .to_crs(epsg=3857).plot(ax=ax, color=color, marker='*', markersize=100)

    try:
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
    except:
        pass

    ax.set_title(f"Episode {ep_idx} - {model_name}")
    ax.legend(loc='upper right')
    ax.axis('off')

    return fig
