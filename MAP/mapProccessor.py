import osmnx as ox
import h3.api.basic_str as h3_api
import geopandas as gpd
import pandas as pd
import pickle
import re
import math
from shapely.geometry import LineString, MultiLineString, Point

# ================= Configuration =================
LOCATION = "London, Ontario, Canada"
H3_RES = 9

ANCHOR_LAT, ANCHOR_LON = 42.995486, -81.253178
ANCHOR_CELL = h3_api.latlng_to_cell(ANCHOR_LAT, ANCHOR_LON, H3_RES)
ANCHOR_IJ = h3_api.cell_to_local_ij(ANCHOR_CELL, ANCHOR_CELL)


def latlon_to_relative_ij(lat, lon):
    """Convert latitude/longitude to relative H3 IJ coordinates."""
    try:
        target_cell = h3_api.latlng_to_cell(lat, lon, H3_RES)
        ij = h3_api.cell_to_local_ij(ANCHOR_CELL, target_cell)
        return (ij[0] - ANCHOR_IJ[0], ij[1] - ANCHOR_IJ[1])
    except Exception as e:
        print(f"Coordinate conversion failed: ({lat}, {lon}) - {e}")
        return None


def clean_maxspeed(speed_val):
    """Normalize road speed values (km/h)."""
    if speed_val is None or str(speed_val) == 'nan':
        return 40.0

    if isinstance(speed_val, list):
        return max([clean_maxspeed(s) for s in speed_val])

    match = re.search(r'\d+', str(speed_val))
    if match:
        val = float(match.group())
        return val * 1.609 if 'mph' in str(speed_val).lower() else val

    return 40.0


def determine_charger_level(row):
    """Heuristically determine EV charger level (L2 or L3)."""

    voltage = str(row.get('voltage', '0')).lower()
    capacity = str(row.get('capacity', '0')).lower()
    operator = str(row.get('operator', '')).lower()
    brand = str(row.get('brand', '')).lower()
    socket = str(row.get('socket', '')).lower()

    l3_keywords = [
        'supercharger', 'tesla', 'chademo', 'ccs', 'combo',
        'fast', 'rapid', 'dc', '50kw', '100kw', '150kw',
        '400v', '800v', 'level3', 'level_3',
        'ivy', 'flo', 'electrify', 'petro-canada', 'onroute'
    ]

    l2_keywords = [
        'level2', 'level_2', 'type2', 'mennekes', 'j1772',
        '22kw', '11kw', '7kw', '3kw', 'ac', 'slow'
    ]

    text_to_check = f"{voltage} {capacity} {operator} {brand} {socket}"

    for keyword in l3_keywords:
        if keyword in text_to_check:
            return "L3"

    for keyword in l2_keywords:
        if keyword in text_to_check:
            return "L2"

    if any(v in voltage for v in ['400', '480', '800', '1000']):
        return "L3"
    elif any(v in voltage for v in ['240', '208', '230', '220']):
        return "L2"

    if 'kw' in capacity:
        try:
            kw = float(re.search(r'(\d+)', capacity).group(1))
            return "L3" if kw >= 50 else "L2"
        except:
            pass

    return "L2"


def charger_matching(chargers_gdf, road_cells):
    """Match EV chargers onto nearest road H3 cells."""

    print(f"Matching {len(chargers_gdf)} chargers to road network...")

    charger_map = {}
    road_ij_list = list(road_cells)

    if len(road_ij_list) > 1000:
        from scipy.spatial import KDTree
        road_points = [(ij[0], ij[1]) for ij in road_ij_list]
        kdtree = KDTree(road_points)

    matched_count = 0

    for idx, row in chargers_gdf.iterrows():

        if idx % 20 == 0:
            print(f"Processing charger {idx}/{len(chargers_gdf)}")

        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        geom_type = geom.geom_type if hasattr(geom, 'geom_type') else geom.type
        points_to_try = []

        if geom_type == 'Point':
            points_to_try.append((geom.y, geom.x))

        elif geom_type == 'Polygon':
            centroid = geom.centroid
            points_to_try.append((centroid.y, centroid.x))
            for x, y in geom.exterior.coords:
                points_to_try.append((y, x))

        elif geom_type == 'MultiPolygon':
            for polygon in geom.geoms:
                centroid = polygon.centroid
                points_to_try.append((centroid.y, centroid.x))

        else:
            centroid = geom.centroid
            points_to_try.append((centroid.y, centroid.x))

        matched = False

        for lat, lon in points_to_try:
            ij = latlon_to_relative_ij(lat, lon)
            if not ij:
                continue

            if ij in road_cells:
                matched = True
            else:
                if len(road_ij_list) > 1000:
                    dist, idx_kd = kdtree.query([ij[0], ij[1]])
                    if dist < 3.0:
                        ij = road_ij_list[idx_kd]
                        matched = True
                else:
                    min_dist = float('inf')
                    nearest_road = None

                    for road_ij in road_ij_list:
                        dist = math.dist(ij, road_ij)
                        if dist < min_dist:
                            min_dist = dist
                            nearest_road = road_ij

                    if min_dist < 10.0 and nearest_road:
                        ij = nearest_road
                        matched = True

            if matched:
                level = determine_charger_level(row)
                charger_map[ij] = level
                matched_count += 1

                if matched_count % 10 == 0:
                    print(f"Matched {matched_count} chargers")

                break

    print(f"Matching finished: {matched_count}/{len(chargers_gdf)} chargers mapped.")
    return charger_map
