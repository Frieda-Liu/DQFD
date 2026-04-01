# improved_map_processor.py
import osmnx as ox
import h3.api.basic_str as h3_api
import geopandas as gpd
import pandas as pd
import pickle
import re
import math
from shapely.geometry import LineString, MultiLineString, Point

# ================= 配置 =================
LOCATION = "London, Ontario, Canada"
H3_RES = 9
ANCHOR_LAT, ANCHOR_LON = 42.995486, -81.253178
ANCHOR_CELL = h3_api.latlng_to_cell(ANCHOR_LAT, ANCHOR_LON, H3_RES)
ANCHOR_IJ = h3_api.cell_to_local_ij(ANCHOR_CELL, ANCHOR_CELL)

def latlon_to_relative_ij(lat, lon):
    """经纬度转相对坐标"""
    try:
        target_cell = h3_api.latlng_to_cell(lat, lon, H3_RES)
        ij = h3_api.cell_to_local_ij(ANCHOR_CELL, target_cell)
        return (ij[0] - ANCHOR_IJ[0], ij[1] - ANCHOR_IJ[1])
    except Exception as e:
        print(f"  坐标转换失败: ({lat}, {lon}) - {e}")
        return None

def clean_maxspeed(speed_val):
    """清理速度数据"""
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
    """智能判断充电桩类型"""
    # 检查各种可能的字段
    voltage = str(row.get('voltage', '0')).lower()
    capacity = str(row.get('capacity', '0')).lower()
    operator = str(row.get('operator', '')).lower()
    brand = str(row.get('brand', '')).lower()
    socket = str(row.get('socket', '')).lower()
    
    # L3的强特征
    # l3_keywords = [
    #     'supercharger', 'tesla', 'chademo', 'ccs', 'combo',
    #     'fast', 'rapid', 'dc', '50kw', '100kw', '150kw',
    #     '400v', '800v', 'level3', 'level_3'
    # ]
    l3_keywords = [
        'supercharger', 'tesla', 'chademo', 'ccs', 'combo',
        'fast', 'rapid', 'dc', '50kw', '100kw', '150kw',
        '400v', '800v', 'level3', 'level_3',
        'ivy', 'flo', 'electrify', 'petro-canada', 'onroute'
    ]
    # L2的特征
    l2_keywords = [
        'level2', 'level_2', 'type2', 'mennekes', 'j1772',
        '22kw', '11kw', '7kw', '3kw', 'ac', 'slow'
    ]
    
    text_to_check = f"{voltage} {capacity} {operator} {brand} {socket}"
    
    # 检查L3关键词
    for keyword in l3_keywords:
        if keyword in text_to_check:
            return "L3"
    
    # 检查L2关键词
    for keyword in l2_keywords:
        if keyword in text_to_check:
            return "L2"
    
    # 默认：根据电压判断
    if any(v in voltage for v in ['400', '480', '800', '1000']):
        return "L3"
    elif any(v in voltage for v in ['240', '208', '230', '220']):
        return "L2"
    
    # 根据容量判断
    if 'kw' in capacity:
        try:
            kw = float(re.search(r'(\d+)', capacity).group(1))
            if kw >= 50:
                return "L3"
            else:
                return "L2"
        except:
            pass
    
    # 默认L2（更常见）
    return "L2"

def charger_matching(chargers_gdf, road_cells):
    """改进的充电桩匹配逻辑"""
    print(f"开始匹配 {len(chargers_gdf)} 个充电桩到道路网络...")
    
    charger_map = {}
    road_ij_list = list(road_cells)
    
    # 构建KD树加速最近邻搜索（如果数据量大）
    if len(road_ij_list) > 1000:
        from scipy.spatial import KDTree
        road_points = [(ij[0], ij[1]) for ij in road_ij_list]
        kdtree = KDTree(road_points)
    
    matched_count = 0
    for idx, row in chargers_gdf.iterrows():
        if idx % 20 == 0:
            print(f"  处理第 {idx}/{len(chargers_gdf)} 个充电桩...")
        
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        
        # 获取几何类型（兼容新旧Shapely）
        if hasattr(geom, 'geom_type'):
            geom_type = geom.geom_type
        else:
            geom_type = geom.type
        
        # 获取位置点
        points_to_try = []
        
        if geom_type == 'Point':
            points_to_try.append((geom.y, geom.x))  # (lat, lon)
        elif geom_type == 'Polygon':
            # 多边形：尝试质心和所有顶点
            centroid = geom.centroid
            points_to_try.append((centroid.y, centroid.x))
            # 添加顶点
            for x, y in geom.exterior.coords:
                points_to_try.append((y, x))
        elif geom_type == 'MultiPolygon':
            # 多边形的每个部分
            for polygon in geom.geoms:
                centroid = polygon.centroid
                points_to_try.append((centroid.y, centroid.x))
        else:
            # 其他几何类型：尝试质心
            centroid = geom.centroid
            points_to_try.append((centroid.y, centroid.x))
        
        # 尝试所有点
        matched = False
        for lat, lon in points_to_try:
            ij = latlon_to_relative_ij(lat, lon)
            if not ij:
                continue
            # 方法1：直接匹配
            if ij in road_cells:
                matched = True
            else:
                # 方法2：寻找最近的道路点
                if len(road_ij_list) > 1000:
                    # 使用KDTree加速
                    dist, idx_kd = kdtree.query([ij[0], ij[1]])
                    if dist < 3.0:  # 3个单位内认为是同一个位置
                        nearest_ij = road_ij_list[idx_kd]
                        ij = nearest_ij
                        matched = True
                else:
                    # 简单线性搜索
                    min_dist = float('inf')
                    nearest_road = None
                    
                    for road_ij in road_ij_list:
                        dist = math.dist(ij, road_ij)
                        if dist < min_dist:
                            min_dist = dist
                            nearest_road = road_ij
                    
                    # 如果距离很近（比如3个单位内），认为是同一个位置
                    if min_dist < 10.0 and nearest_road:
                        ij = nearest_road
                        matched = True
            
            if matched:
                # 判断充电类型
                level = determine_charger_level(row)
                charger_map[ij] = level
                matched_count += 1
                
                if matched_count % 10 == 0:
                    print(f"    已匹配 {matched_count} 个充电桩")
                break
    
    print(f"匹配完成：成功匹配 {matched_count}/{len(chargers_gdf)} 个充电桩")
    return charger_map

def collect_road_network():
    print("收集道路网络数据...")
    
    # 抓取路网
    G_osm = ox.graph_from_place(LOCATION, network_type="drive")
    edges = ox.graph_to_gdfs(G_osm, nodes=False, edges=True)
    
    # 过滤主干道
    main_types = ['motorway', 'trunk', 'primary', 'secondary', 'tertiary']
    edges = edges[edges['highway'].apply(
        lambda x: any(t in main_types for t in (x if isinstance(x, list) else [x]))
    )]
    
    # 处理道路格子
    road_cells = set()
    speed_data = {}
    
    print("进行H3网格化...")
    for counter, (idx, road) in enumerate(edges.iterrows()): 
        if counter % 100 == 0: # 用 counter 取余
            print(f"  处理第 {counter}/{len(edges)} 条道路...")
        
        speed = clean_maxspeed(road.get('maxspeed'))
        geom = road.geometry
        
        # 获取所有坐标点
        coords = []
        if hasattr(geom, 'geom_type'):
            geom_type = geom.geom_type
        else:
            geom_type = geom.type
        
        if geom_type == 'LineString':
            coords = list(geom.coords)
        elif geom_type == 'MultiLineString':
            for line in geom.geoms:
                coords.extend(list(line.coords))
        
        # 处理每个坐标点
        for lon, lat in coords:
            ij = latlon_to_relative_ij(lat, lon)
            if ij:
                road_cells.add(ij)
                if ij not in speed_data:
                    speed_data[ij] = []
                speed_data[ij].append(speed)
    
    # 计算平均速度
    speed_map = {}
    for ij, speeds in speed_data.items():
        if speeds:
            speed_map[ij] = sum(speeds) / len(speeds)
    
    print(f"道路网络处理完成：{len(road_cells)} 个道路格子")
    return road_cells, speed_map

def collect_chargers(road_cells):
    """收集充电桩数据"""
    print("收集充电桩数据...")
    
    # 多种充电桩标签
    charger_tags = [
        {"amenity": "charging_station"},
        {"highway": "ev_charging"},
        {"amenity": "fuel", "fuel:EV": True},
        {"amenity": "fuel", "fuel:electric": True},
        {"ev_charging": "*"},
        {"charge": "*"},  # 通配符
    ]
    
    all_chargers = []
    
    for tags in charger_tags:
        try:
            chargers = ox.features_from_place(LOCATION, tags=tags)
            if not chargers.empty:
                print(f"  找到标签 {tags}: {len(chargers)} 个点位")
                all_chargers.append(chargers)
        except Exception as e:
            print(f"  标签 {tags} 查询失败: {e}")
            continue
    
    if all_chargers:
        # 合并并去重
        chargers_gdf = pd.concat(all_chargers, ignore_index=True)
        # 基于几何位置去重
        chargers_gdf = chargers_gdf.drop_duplicates(subset=['geometry'])
        print(f"去重后充电桩数量: {len(chargers_gdf)}")
    else:
        chargers_gdf = gpd.GeoDataFrame()
        print("警告：未找到任何充电桩数据")
    
    # 改进的匹配
    charger_map = charger_matching(chargers_gdf, road_cells)
    
    return charger_map

def collect_traffic_signals(road_cells):
    """收集交通信号数据"""
    print("收集交通信号数据...")
    
    try:
        signals = ox.features_from_place(LOCATION, tags={"highway": "traffic_signals"})
        print(f"找到 {len(signals)} 个交通信号")
        
        signal_cells = set()
        matched_count = 0
        
        for counter, (idx, row) in enumerate(signals.iterrows()):
            if counter % 50 == 0:
                print(f"  处理第 {counter}/{len(signals)} 个交通信号...")
            
            geom = row.geometry
            if geom is None:
                continue
            
            # 获取几何类型
            if hasattr(geom, 'geom_type'):
                geom_type = geom.geom_type
            else:
                geom_type = geom.type
            
            if geom_type == 'Point':
                lon, lat = geom.x, geom.y
                ij = latlon_to_relative_ij(lat, lon)
                
                if ij:
                    # 检查是否在道路上或附近
                    if ij in road_cells:
                        signal_cells.add(ij)
                        matched_count += 1
                    else:
                        # 寻找最近的道路点
                        min_dist = float('inf')
                        nearest_road = None
                        
                        for road_ij in road_cells:
                            dist = math.dist(ij, road_ij)
                            if dist < min_dist:
                                min_dist = dist
                                nearest_road = road_ij
                        
                        # 如果距离很近，认为是同一个位置
                        if min_dist < 2.0 and nearest_road:
                            signal_cells.add(nearest_road)
                            matched_count += 1
        
        print(f"交通信号匹配完成：{matched_count}/{len(signals)} 个匹配到道路")
        return signal_cells
        
    except Exception as e:
        print(f"交通信号收集失败: {e}")
        return set()

def main():
    """主函数"""
    print(f"开始处理 {LOCATION} 的地图数据...")
    
    # 1. 收集道路网络
    road_cells, speed_map = collect_road_network()
    
    # 2. 收集充电桩
    charger_map = collect_chargers(road_cells)
    
    # 3. 收集交通信号
    signal_cells = collect_traffic_signals(road_cells)
    
    # 4. 分析结果
    print("\n=== 数据收集结果 ===")
    print(f"道路格子数量: {len(road_cells)}")
    print(f"充电桩数量: {len(charger_map)}")
    print(f"交通信号数量: {len(signal_cells)}")
    
    # 充电桩类型分析
    if charger_map:
        l2_count = sum(1 for v in charger_map.values() if v == "L2")
        l3_count = sum(1 for v in charger_map.values() if v == "L3")
        print(f"充电桩类型分布:")
        print(f"  L2: {l2_count} ({l2_count/len(charger_map)*100:.1f}%)")
        print(f"  L3: {l3_count} ({l3_count/len(charger_map)*100:.1f}%)")
    
    # 5. 保存数据
    output = {
        "road_cells": road_cells,
        "speed_map": speed_map,
        "traffic_signals": signal_cells,
        "chargers": charger_map,
        "config": {
            "res": H3_RES,
            "anchor_latlng": (ANCHOR_LAT, ANCHOR_LON),
            "anchor_ij": ANCHOR_IJ,
            "location": LOCATION
        }
    }
    
    # 保存两份，一份清理版，一份原始版
    with open("london_data_improved.pkl", "wb") as f:
        pickle.dump(output, f)
    
    # 同时保存原始数据用于调试
    output_debug = {
        "road_cells": road_cells,
        "speed_map": speed_map,
        "chargers": charger_map,
        "traffic_signals": signal_cells,
        "config": output["config"],
        "stats": {
            "road_cell_count": len(road_cells),
            "charger_count": len(charger_map),
            "signal_count": len(signal_cells)
        }
    }
    
    with open("london_data_debug.pkl", "wb") as f:
        pickle.dump(output_debug, f)
    
    print(f"\n数据已保存:")
    print(f"  london_data_improved.pkl - 主要数据文件")
    print(f"  london_data_debug.pkl - 调试用数据文件")
    
    # 覆盖率分析
    if road_cells and charger_map:
        # 抽样计算平均覆盖距离
        sample_size = min(200, len(road_cells))
        sample_roads = list(road_cells)[:sample_size]
        
        total_dist = 0
        for road_ij in sample_roads:
            min_dist = min(math.dist(road_ij, charger_ij) for charger_ij in charger_map.keys())
            total_dist += min_dist
        
        avg_dist = total_dist / sample_size
        print(f"平均到最近充电站距离: {avg_dist:.2f} 单位")
        
        if avg_dist > 15:
            print("⚠️  覆盖距离较大，建议增加充电桩密度")
        elif avg_dist > 10:
            print("✅ 覆盖距离适中")
        else:
            print("🎉 覆盖距离很好")

if __name__ == "__main__":
    main()