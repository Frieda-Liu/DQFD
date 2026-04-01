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

# 导入您的自定义模块
from mutilEnv import HexTrafficEnv  
from mutilDqfsAgent import ExpertDQN 

# --- 1. 路径与 H3 配置 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 模型存放文件夹 (建议在 Train 目录下新建一个 models 文件夹)
MODEL_FOLDER = os.path.join(BASE_DIR, "models")
if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)

# 伦敦锚点配置 (需与 map_processor 一致)
ANCHOR_LAT, ANCHOR_LON = 42.995486, -81.253178
H3_RES = 9
ANCHOR_CELL = h3_api.latlng_to_cell(ANCHOR_LAT, ANCHOR_LON, H3_RES)

# --- 2. 坐标转换工具函数 ---
def get_h3_polygon(i, j):
    """IJ -> H3 Hexagon Polygon"""
    try:
        anchor_ij = h3_api.cell_to_local_ij(ANCHOR_CELL, ANCHOR_CELL)
        target_i = i + anchor_ij[0]
        target_j = j + anchor_ij[1]
        target_cell = h3_api.local_ij_to_cell(ANCHOR_CELL, target_i, target_j)
        boundary = h3_api.cell_to_boundary(target_cell)
        return Polygon([(lon, lat) for lat, lon in boundary])
    except: return None

def get_latlon_point(i, j):
    """IJ -> (lon, lat) Point"""
    try:
        anchor_ij = h3_api.cell_to_local_ij(ANCHOR_CELL, ANCHOR_CELL)
        target_i = i + anchor_ij[0]
        target_j = j + anchor_ij[1]
        target_cell = h3_api.local_ij_to_cell(ANCHOR_CELL, target_i, target_j)
        lat, lon = h3_api.cell_to_latlng(target_cell)
        return lon, lat
    except: return None

# --- 3. 核心绘图函数 (包含轨迹) ---
def draw_analysis_map(env, model_name):
    st.subheader(f"🗺️ Trajectory Analysis: {model_name}")
    
    with st.spinner("Generating geographic visualization..."):
        # A. 基础背景 (路网与充电桩)
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

        # C. 最终位置
        final_pts = [Point(get_latlon_point(pos[0], pos[1])) for pos in env.agent_positions if get_latlon_point(pos[0], pos[1])]
        gdf_agents = gpd.GeoDataFrame(geometry=final_pts, crs="EPSG:4326")

        # D. Matplotlib 渲染
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 转换为 Web Mercator (3857)
        if not gdf_road.empty:
            gdf_road.to_crs(epsg=3857).plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=0.2, alpha=0.1)
        
        # 绘制充电桩
        if l2_list:
            gpd.GeoDataFrame(geometry=[p.centroid for p in l2_list], crs="EPSG:4326").to_crs(epsg=3857).plot(
                ax=ax, color='cyan', marker='+', markersize=40, label='L2 Charger', alpha=0.6)
        if l3_list:
            gpd.GeoDataFrame(geometry=[p.centroid for p in l3_list], crs="EPSG:4326").to_crs(epsg=3857).plot(
                ax=ax, color='red', marker='P', markersize=80, label='L3 Fast Charger', alpha=0.8)
        
        # 绘制历史轨迹 (橙色线条)
        if not gdf_tracks.empty:
            gdf_tracks.to_crs(epsg=3857).plot(ax=ax, color='orange', linewidth=1, alpha=0.4, label='Path History')
        
        # 绘制最终停留点 (黄色圆点)
        if not gdf_agents.empty:
            gdf_agents.to_crs(epsg=3857).plot(ax=ax, color='yellow', marker='o', markersize=35, edgecolor='black', zorder=5)

        # 添加底图
        try:
            cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
        except:
            st.warning("Basemap server busy, showing only coordinates.")

        ax.set_axis_off()
        ax.legend(loc='upper right', fontsize='x-small', framealpha=0.5)
        st.pyplot(fig)

# --- 4. 页面基础设置 ---
st.set_page_config(page_title="EV RL Testbed", layout="wide", page_icon="🔋")
st.title("🔋 London EV Multi-Agent Scheduling Platform")

# --- 5. 侧边栏：模型选择与配置 ---
with st.sidebar:
    st.header("📁 Model Repository")
    # 自动扫描 .pth 文件
    model_list = [f for f in os.listdir(MODEL_FOLDER) if f.endswith('.pth')]
    
    if not model_list:
        st.error(f"Please put .pth models in: {MODEL_FOLDER}")
        selected_model = None
    else:
        selected_model = st.selectbox("Select Model Version:", options=model_list)
        st.info(f"Active: {selected_model}")

    st.divider()
    st.header("⚙️ Evaluation Settings")
    num_agents = st.slider("Agents", 1, 20, 10)
    soc_limit = st.slider("Charging Threshold (SoC %)", 5.0, 50.0, 25.0)
    expert_w = st.slider("Expert Guidance weight", 0.0, 1.0, 0.1)
    weather_val = st.select_slider("Weather", options=[0.8, 1.0, 1.2, 1.5, 2.0], value=1.2)
    test_episodes = st.number_input("Episodes", min_value=1, max_value=50, value=5)
    
    st.divider()
    run_button = st.button("🚀 Run Simulation", use_container_width=True)

# --- 6. 资源加载 (根据模型名缓存) ---
@st.cache_resource
def load_eval_assets(_num_agents, model_name):
    env = HexTrafficEnv(num_agents=_num_agents)
    agent = ExpertDQN(state_dim=20, action_dim=6)
    if model_name:
        path = os.path.join(MODEL_FOLDER, model_name)
        try:
            agent.policy_net.load_state_dict(torch.load(path, map_location='cpu'))
            st.sidebar.success("✅ Neural weights synced")
        except Exception as e:
            st.sidebar.error(f"❌ Load error: {e}")
    return env, agent

# 初始化环境与模型
env, agent = load_eval_assets(num_agents, selected_model)
env.soc_threshold = soc_limit
env.weather_factor = weather_val

# --- 7. 运行模拟逻辑 ---
if run_button and selected_model:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    stats = {
        "success": 0, "out_of_battery": 0, "out_of_road": 0, "timeout": 0,
        "final_soc": [], "total_reward": 0
    }

    for ep in range(int(test_episodes)):
        status_text.text(f"Processing Episode {ep+1}/{test_episodes}...")
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            actions = []
            for i in range(num_agents):
                agent.epsilon = 0.0 # 推理模式
                act = agent.select_action(obs[i], env, agent_id=i, training=True, expert_weight=expert_w)
                actions.append(act)
            
            obs, rewards, all_done, truncated, infos = env.step(actions)
            ep_reward += sum(rewards) / num_agents
            done = all_done or truncated
        
        # 统计本局 ASR (Individual Success Rate)
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

    status_text.success(f"✅ Evaluation complete using {selected_model}")

    # --- 8. 结果看板 ---
    total_samples = test_episodes * num_agents
    sr = (stats["success"] / total_samples) * 100
    avg_soc = np.mean(stats["final_soc"]) if stats["final_soc"] else 0

    m1, m2, m3 = st.columns(3)
    m1.metric("Individual Success Rate (ASR)", f"{sr:.2f}%", delta=f"{stats['success']}/{total_samples}")
    m2.metric("Avg Remaining SoC", f"{avg_soc:.2f}%")
    m3.metric("Avg Reward / Episode", f"{stats['total_reward']/test_episodes:.2f}")

    st.divider()

    # --- 9. 地图展示 (包含轨迹历史) ---
    draw_analysis_map(env, selected_model)

    st.divider()

    # --- 10. 失败原因分布图 ---
    st.subheader("📌 Performance Breakdown")
    chart_data = pd.DataFrame({
        "Outcome": ["Success", "No Battery", "Timeout", "Crashed"],
        "Count": [stats["success"], stats["out_of_battery"], stats["timeout"], stats["out_of_road"]]
    })
    st.bar_chart(chart_data.set_index("Outcome"))

    with st.expander("Show detailed stats JSON"):
        st.write(stats)
else:
    st.info("👈 Please select a model from the repository and set parameters, then click 'Run Simulation'.")