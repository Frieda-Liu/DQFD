import streamlit as st
import pandas as pd
import torch
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import h3.api.basic_str as h3_api
import contextily as cx
from shapely.geometry import Polygon, Point, LineString
import geopandas as gpd

# 导入自定义模块
from mutilEnv import HexTrafficEnv  
from mutilDqfsAgent import ExpertDQN 

# --- 1. 全局配置 (与 map_processor 保持一致) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILENAME = "refined_final_model.pth" 
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)

ANCHOR_LAT, ANCHOR_LON = 42.995486, -81.253178
H3_RES = 9
ANCHOR_CELL = h3_api.latlng_to_cell(ANCHOR_LAT, ANCHOR_LON, H3_RES)

# --- 2. 坐标转换工具函数 ---
def get_h3_polygon(i, j):
    """将相对坐标 IJ 转回 H3 六边形 Polygon"""
    try:
        anchor_ij = h3_api.cell_to_local_ij(ANCHOR_CELL, ANCHOR_CELL)
        target_i = i + anchor_ij[0]
        target_j = j + anchor_ij[1]
        target_cell = h3_api.local_ij_to_cell(ANCHOR_CELL, target_i, target_j)
        boundary = h3_api.cell_to_boundary(target_cell)
        return Polygon([(lon, lat) for lat, lon in boundary])
    except:
        return None

def get_latlon_point(i, j):
    """将相对坐标 IJ 转回 (lon, lat) 点"""
    try:
        anchor_ij = h3_api.cell_to_local_ij(ANCHOR_CELL, ANCHOR_CELL)
        target_i = i + anchor_ij[0]
        target_j = j + anchor_ij[1]
        target_cell = h3_api.local_ij_to_cell(ANCHOR_CELL, target_i, target_j)
        lat, lon = h3_api.cell_to_latlng(target_cell)
        return lon, lat
    except:
        return None

# --- 3. 轨迹地图绘制函数 ---
def draw_trajectory_map(env):
    st.subheader("🗺️ Infrastructure & Agent Trajectory Monitor")
    
    with st.spinner("Rendering geographic data..."):
        # A. 背景：路网与充电站
        road_polys = [get_h3_polygon(ij[0], ij[1]) for ij in env.london_main_roads]
        gdf_road = gpd.GeoDataFrame(geometry=[p for p in road_polys if p], crs="EPSG:4326")
        
        l2_list = [get_h3_polygon(ij[0], ij[1]) for ij, lv in env.charging_stations.items() if lv == 'L2']
        l3_list = [get_h3_polygon(ij[0], ij[1]) for ij, lv in env.charging_stations.items() if lv == 'L3']
        
        # B. 轨迹线 (Trajectories)
        lines = []
        if hasattr(env, 'trajectories'):
            for track in env.trajectories:
                pts = [get_latlon_point(p[0], p[1]) for p in track]
                valid_pts = [p for p in pts if p is not None]
                if len(valid_pts) >= 2:
                    lines.append(LineString(valid_pts))
        gdf_tracks = gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326")

        # C. 最终位置 (Final Positions)
        final_pts = []
        for pos in env.agent_positions:
            lon_lat = get_latlon_point(pos[0], pos[1])
            if lon_lat: final_pts.append(Point(lon_lat))
        gdf_agents = gpd.GeoDataFrame(geometry=final_pts, crs="EPSG:4326")

        # D. Matplotlib 绘图
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 转换投影为 Web Mercator (3857)
        if not gdf_road.empty:
            gdf_road.to_crs(epsg=3857).plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=0.2, alpha=0.1)
        
        # 绘制充电桩
        if l2_list:
            gpd.GeoDataFrame(geometry=[p.centroid for p in l2_list], crs="EPSG:4326").to_crs(epsg=3857).plot(
                ax=ax, color='cyan', marker='+', markersize=40, label='L2 Charger', alpha=0.6
            )
        if l3_list:
            gpd.GeoDataFrame(geometry=[p.centroid for p in l3_list], crs="EPSG:4326").to_crs(epsg=3857).plot(
                ax=ax, color='red', marker='P', markersize=80, label='L3 Fast Charger', alpha=0.8
            )
        
        # 绘制轨迹
        if not gdf_tracks.empty:
            gdf_tracks.to_crs(epsg=3857).plot(ax=ax, color='orange', linewidth=1, alpha=0.5, label='Agent Trajectories')
        
        # 绘制最终位置
        if not gdf_agents.empty:
            gdf_agents.to_crs(epsg=3857).plot(ax=ax, color='yellow', marker='o', markersize=30, edgecolor='black', zorder=5)

        # 添加底图
        try:
            cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
        except:
            st.warning("Could not load basemap tiles.")

        ax.set_axis_off()
        ax.legend(loc='upper right', fontsize='x-small')
        st.pyplot(fig)

# --- 4. 页面基础设置 ---
st.set_page_config(page_title="EV RL Simulator", layout="wide", page_icon="🔋")
st.title("🔋 London EV Charging Scheduling Platform")

# --- 5. 资源加载 (带缓存) ---
@st.cache_resource
def load_assets(_num_agents):
    env = HexTrafficEnv(num_agents=_num_agents)
    agent = ExpertDQN(state_dim=20, action_dim=6) 
    if os.path.exists(MODEL_PATH):
        try:
            agent.policy_net.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
            st.sidebar.success(f"✅ Loaded weights: {MODEL_FILENAME}")
        except:
            st.sidebar.error("❌ Weight loading failed.")
    return env, agent

# --- 6. 侧边栏交互 ---
with st.sidebar:
    st.header("⚙️ Settings")
    num_agents = st.slider("Number of Agents", 1, 20, 10)
    soc_limit = st.slider("Charging Threshold (SoC %)", 5.0, 50.0, 25.0)
    expert_w = st.slider("Expert Guidance Weight", 0.0, 1.0, 0.1)
    weather_val = st.select_slider("Weather Factor", options=[0.8, 1.0, 1.2, 1.5, 2.0], value=1.2)
    test_episodes = st.number_input("Episodes", min_value=1, max_value=100, value=10)
    st.divider()
    run_button = st.button("🚀 Run Evaluation", use_container_width=True)

env, agent = load_assets(num_agents)
env.soc_threshold = soc_limit
env.weather_factor = weather_val

# --- 7. 运行模拟逻辑 ---
if run_button:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 统计数据初始化
    stats = {
        "success": 0, "out_of_battery": 0, "out_of_road": 0, "timeout": 0,
        "final_soc": [], "total_reward": 0
    }

    for ep in range(int(test_episodes)):
        status_text.text(f"Running Episode {ep+1}/{test_episodes}...")
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            actions = []
            for i in range(num_agents):
                # 评估时关闭 epsilon 随机性，仅使用专家权重干预
                agent.epsilon = 0.0 
                act = agent.select_action(obs[i], env, agent_id=i, training=True, expert_weight=expert_w)
                actions.append(act)
            
            obs, rewards, all_done, truncated, infos = env.step(actions)
            ep_reward += sum(rewards) / num_agents
            done = all_done or truncated
        
        # 统计本局车辆状态 (对齐您的 ASR 逻辑)
        for v in env.vehicles:
            if v.goal:
                stats["success"] += 1
                stats["final_soc"].append(v.soc)
            elif v.finish_status == "out_of_battery":
                stats["out_of_battery"] += 1
            elif v.finish_status == "out_of_road":
                stats["out_of_road"] += 1
            else:
                stats["timeout"] += 1
        
        stats["total_reward"] += ep_reward
        progress_bar.progress((ep + 1) / test_episodes)

    status_text.success("✅ Simulation Finished!")

    # --- 8. 结果看板 ---
    total_samples = test_episodes * num_agents
    success_rate = (stats["success"] / total_samples) * 100
    avg_soc = np.mean(stats["final_soc"]) if stats["final_soc"] else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Individual Success Rate", f"{success_rate:.2f}%", delta=f"{stats['success']}/{total_samples}")
    col2.metric("Avg Final SoC", f"{avg_soc:.2f}%")
    col3.metric("Avg Episode Reward", f"{stats['total_reward']/test_episodes:.2f}")

    st.divider()

    # --- 9. 地图展示 (包含轨迹) ---
    draw_trajectory_map(env)

    st.divider()

    # --- 10. 任务状态分布图 ---
    st.subheader("📌 Task Outcome Distribution")
    res_df = pd.DataFrame({
        "Status": ["Success", "Battery Empty", "Timeout", "Crashed"],
        "Count": [stats["success"], stats["out_of_battery"], stats["timeout"], stats["out_of_road"]]
    })
    st.bar_chart(res_df.set_index("Status"))

    with st.expander("Show raw data JSON"):
        st.write(stats)

else:
    st.info("👈 Set evaluation parameters in the sidebar and click 'Run Evaluation'.")