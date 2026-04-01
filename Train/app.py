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

# 导入你的自定义模块
from mutilEnv import HexTrafficEnv  
from mutilDqfsAgent import ExpertDQN 

# --- 1. 核心配置 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 建议在 Train 目录下创建一个 models 文件夹存放所有 .pth 文件
MODEL_FOLDER = os.path.join(BASE_DIR, "models")
if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)

# 伦敦 H3 锚点配置 (必须与 map_processor 一致)
ANCHOR_LAT, ANCHOR_LON = 42.995486, -81.253178
H3_RES = 9
ANCHOR_CELL = h3_api.latlng_to_cell(ANCHOR_LAT, ANCHOR_LON, H3_RES)

# --- 2. 坐标转换工具 ---
def get_h3_polygon(i, j):
    """相对 IJ 转 H3 六边形"""
    try:
        anchor_ij = h3_api.cell_to_local_ij(ANCHOR_CELL, ANCHOR_CELL)
        target_i = i + anchor_ij[0]
        target_j = j + anchor_ij[1]
        target_cell = h3_api.local_ij_to_cell(ANCHOR_CELL, target_i, target_j)
        boundary = h3_api.cell_to_boundary(target_cell)
        return Polygon([(lon, lat) for lat, lon in boundary])
    except: return None

def get_latlon_point(i, j):
    """相对 IJ 转 (经度, 纬度)"""
    try:
        anchor_ij = h3_api.cell_to_local_ij(ANCHOR_CELL, ANCHOR_CELL)
        target_i = i + anchor_ij[0]
        target_j = j + anchor_ij[1]
        target_cell = h3_api.local_ij_to_cell(ANCHOR_CELL, target_i, target_j)
        lat, lon = h3_api.cell_to_latlng(target_cell)
        return lon, lat
    except: return None

# --- 3. 多色轨迹地图绘制 ---
def draw_analysis_map(env, model_name):
    st.subheader(f"🗺️ Multi-Agent Trajectory Analysis ({model_name})")
    
    with st.spinner("Generating London map with agent paths..."):
        # 准备背景图层
        road_polys = [get_h3_polygon(ij[0], ij[1]) for ij in env.london_main_roads]
        gdf_road = gpd.GeoDataFrame(geometry=[p for p in road_polys if p], crs="EPSG:4326")
        
        l2_list = [get_h3_polygon(ij[0], ij[1]) for ij, lv in env.charging_stations.items() if lv == 'L2']
        l3_list = [get_h3_polygon(ij[0], ij[1]) for ij, lv in env.charging_stations.items() if lv == 'L3']
        
        # 为每个 Agent 分配独立颜色
        colors = plt.cm.rainbow(np.linspace(0, 1, env.num_agents))

        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 1. 绘制路网
        if not gdf_road.empty:
            gdf_road.to_crs(epsg=3857).plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=0.2, alpha=0.1)
        
        # 2. 绘制充电桩
        if l2_list:
            gpd.GeoDataFrame(geometry=[p.centroid for p in l2_list], crs="EPSG:4326").to_crs(epsg=3857).plot(
                ax=ax, color='cyan', marker='+', markersize=40, label='L2 Charger', alpha=0.5)
        if l3_list:
            gpd.GeoDataFrame(geometry=[p.centroid for p in l3_list], crs="EPSG:4326").to_crs(epsg=3857).plot(
                ax=ax, color='red', marker='P', markersize=80, label='L3 Fast Charger', alpha=0.7)
        
        # 3. 按 Agent 绘制轨迹和终点
        if hasattr(env, 'trajectories'):
            for i in range(env.num_agents):
                c = colors[i]
                # 绘制历史轨迹线
                track = env.trajectories[i]
                pts = [get_latlon_point(p[0], p[1]) for p in track]
                valid_pts = [p for p in pts if p is not None]
                if len(valid_pts) >= 2:
                    gpd.GeoDataFrame(geometry=[LineString(valid_pts)], crs="EPSG:4326").to_crs(epsg=3857).plot(
                        ax=ax, color=c, linewidth=1.5, alpha=0.6, label=f"Agent {i}" if env.num_agents <= 8 else "")
                
                # 绘制终点圆点
                final_pos = env.agent_positions[i]
                lon_lat = get_latlon_point(final_pos[0], final_pos[1])
                if lon_lat:
                    gpd.GeoDataFrame(geometry=[Point(lon_lat)], crs="EPSG:4326").to_crs(epsg=3857).plot(
                        ax=ax, color=c, marker='o', markersize=40, edgecolor='black', zorder=5)

        # 4. 添加底图
        try:
            cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
        except:
            st.warning("Basemap server connection failed, showing vector grid.")

        ax.set_axis_off()
        if env.num_agents <= 10:
            ax.legend(loc='upper right', fontsize='xx-small', ncol=2)
        st.pyplot(fig)

# --- 4. 侧边栏与模型加载 ---
st.set_page_config(page_title="EV Multi-Agent Platform", layout="wide", page_icon="🔋")
st.title("🔋 London EV Multi-Agent Scheduling Simulator")

with st.sidebar:
    st.header("📂 Model Repository")
    all_models = [f for f in os.listdir(MODEL_FOLDER) if f.endswith('.pth')]
    selected_model = st.selectbox("Select Model Version:", options=all_models) if all_models else None
    
    st.divider()
    st.header("⚙️ Simulation Settings")
    num_agents = st.slider("Number of Agents", 1, 20, 10)
    soc_limit = st.slider("Charging Threshold (SoC %)", 5.0, 50.0, 25.0)
    expert_w = st.slider("Expert Guidance Weight", 0.0, 1.0, 0.1)
    weather_val = st.select_slider("Weather Factor", options=[0.8, 1.0, 1.2, 1.5, 2.0], value=1.2)
    test_episodes = st.number_input("Episodes", min_value=1, max_value=100, value=5)
    
    st.divider()
    run_button = st.button("🚀 Start Evaluation", use_container_width=True)

@st.cache_resource
def load_eval_assets(_num_agents, model_name):
    # 只要 _num_agents 变了，环境就会重新创建，防止维度冲突
    env = HexTrafficEnv(num_agents=_num_agents)
    agent = ExpertDQN(state_dim=20, action_dim=6)
    if model_name:
        path = os.path.join(MODEL_FOLDER, model_name)
        agent.policy_net.load_state_dict(torch.load(path, map_location='cpu'))
    return env, agent

if not selected_model:
    st.warning("Please upload .pth models to the /models folder.")
    st.stop()

# 加载环境和模型
env, agent = load_eval_assets(num_agents, selected_model)
env.soc_threshold = soc_limit
env.weather_factor = weather_val

# --- 5. 核心仿真循环 ---
if run_button:
    progress_bar = st.progress(0)
    status_text = st.empty()
    stats = {"success": 0, "out_of_battery": 0, "out_of_road": 0, "timeout": 0, "final_soc": [], "total_reward": 0}

    for ep in range(int(test_episodes)):
        status_text.text(f"Running Episode {ep+1}/{test_episodes}...")
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            actions = []
            for i in range(num_agents):
                agent.epsilon = 0.0 # 评估模式
                act = agent.select_action(obs[i], env, agent_id=i, training=True, expert_weight=expert_w)
                actions.append(int(act))
            
            # 使用 list() 确保传递的是列表，解决单智能体报错
            obs, rewards, all_done, truncated, infos = env.step(list(actions))
            ep_reward += sum(rewards) / num_agents
            done = all_done or truncated
        
        # 统计本局 ASR
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

    status_text.success(f"✅ Simulation Complete using {selected_model}")

    # --- 6. 结果展示看板 ---
    total_samples = test_episodes * num_agents
    sr = (stats["success"] / total_samples) * 100
    avg_soc = np.mean(stats["final_soc"]) if stats["final_soc"] else 0

    m1, m2, m3 = st.columns(3)
    m1.metric("Individual Success Rate (ASR)", f"{sr:.2f}%", f"{stats['success']}/{total_samples}")
    m2.metric("Avg Final SoC", f"{avg_soc:.2f}%")
    m3.metric("Avg Reward / Episode", f"{stats['total_reward']/test_episodes:.2f}")

    st.divider()
    # 绘制多色轨迹地图
    draw_analysis_map(env, selected_model)
    st.divider()

    # 失败原因图表
    st.subheader("📊 Outcome Performance Breakdown")
    chart_data = pd.DataFrame({
        "Result": ["Success", "No Battery", "Timeout", "Crashed"],
        "Count": [stats["success"], stats["out_of_battery"], stats["timeout"], stats["out_of_road"]]
    })
    st.bar_chart(chart_data.set_index("Result"))

else:
    st.info("👈 Select a model file and adjust parameters, then click 'Start Evaluation'.")