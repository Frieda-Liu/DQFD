import streamlit as st
import pandas as pd
import torch
import numpy as np
import os
import time
import copy
import matplotlib.pyplot as plt
import h3.api.basic_str as h3_api
import contextily as cx
from shapely.geometry import Polygon, Point, LineString
import geopandas as gpd

# 导入自定义模块
from mutilEnv import HexTrafficEnv  
from mutilDqfsAgent import ExpertDQN 

# --- 1. 基础配置 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.join(BASE_DIR, "models")
if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)

ANCHOR_LAT, ANCHOR_LON = 42.995486, -81.253178
H3_RES = 9
ANCHOR_CELL = h3_api.latlng_to_cell(ANCHOR_LAT, ANCHOR_LON, H3_RES)

# --- 2. 坐标转换工具 ---
def get_h3_polygon(i, j):
    try:
        anchor_ij = h3_api.cell_to_local_ij(ANCHOR_CELL, ANCHOR_CELL)
        target_i = i + anchor_ij[0]
        target_j = j + anchor_ij[1]
        target_cell = h3_api.local_ij_to_cell(ANCHOR_CELL, target_i, target_j)
        boundary = h3_api.cell_to_boundary(target_cell)
        return Polygon([(lon, lat) for lat, lon in boundary])
    except: return None

def get_latlon_point(i, j):
    try:
        anchor_ij = h3_api.cell_to_local_ij(ANCHOR_CELL, ANCHOR_CELL)
        target_i = i + anchor_ij[0]
        target_j = j + anchor_ij[1]
        target_cell = h3_api.local_ij_to_cell(ANCHOR_CELL, target_i, target_j)
        lat, lon = h3_api.cell_to_latlng(target_cell)
        return lon, lat
    except: return None

# --- 3. 绘图函数 (单局回放渲染) ---
def draw_replay_map(env, tracks_to_draw, agent_positions, model_name, ep_idx):
    st.subheader(f"📍 Replaying Episode {ep_idx} ({model_name})")
    
    with st.spinner(f"Rendering Episode {ep_idx}..."):
        # 基础路网和充电站 (仅需一次转换，但在 streamlit 重绘时为了稳妥重新计算)
        road_polys = [get_h3_polygon(ij[0], ij[1]) for ij in env.london_main_roads]
        gdf_road = gpd.GeoDataFrame(geometry=[p for p in road_polys if p], crs="EPSG:4326")
        l2_list = [get_h3_polygon(ij[0], ij[1]) for ij, lv in env.charging_stations.items() if lv == 'L2']
        l3_list = [get_h3_polygon(ij[0], ij[1]) for ij, lv in env.charging_stations.items() if lv == 'L3']
        
        num_agents = len(tracks_to_draw)
        colors = plt.cm.rainbow(np.linspace(0, 1, num_agents))
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 绘制背景
        if not gdf_road.empty:
            gdf_road.to_crs(epsg=3857).plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=0.1, alpha=0.05)
        
        if l2_list:
            gpd.GeoDataFrame(geometry=[p.centroid for p in l2_list], crs="EPSG:4326").to_crs(epsg=3857).plot(
                ax=ax, color='cyan', marker='+', markersize=40, label='L2 Charger', alpha=0.4)
        if l3_list:
            gpd.GeoDataFrame(geometry=[p.centroid for p in l3_list], crs="EPSG:4326").to_crs(epsg=3857).plot(
                ax=ax, color='red', marker='P', markersize=80, label='L3 Fast Charger', alpha=0.6)

        # 按 Agent 绘制这一局的轨迹
        for i in range(num_agents):
            c = colors[i]
            track = tracks_to_draw[i]
            pts = [get_latlon_point(p[0], p[1]) for p in track]
            valid_pts = [p for p in pts if p is not None]
            
            if len(valid_pts) >= 2:
                # 轨迹线
                gpd.GeoDataFrame(geometry=[LineString(valid_pts)], crs="EPSG:4326").to_crs(epsg=3857).plot(
                    ax=ax, color=c, linewidth=2, alpha=0.7, label=f"Agent {i}" if num_agents <= 8 else "")
                # 起点
                gpd.GeoDataFrame(geometry=[Point(valid_pts[0])], crs="EPSG:4326").to_crs(epsg=3857).plot(
                    ax=ax, color=c, marker='o', markersize=20, edgecolor='white')
                # 终点 (带星号)
                gpd.GeoDataFrame(geometry=[Point(valid_pts[-1])], crs="EPSG:4326").to_crs(epsg=3857).plot(
                    ax=ax, color=c, marker='*', markersize=100, edgecolor='black', zorder=5)

        try:
            cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
        except: pass
        
        ax.set_axis_off()
        if num_agents <= 10:
            ax.legend(loc='upper right', fontsize='xx-small', ncol=2)
        st.pyplot(fig)

# --- 4. 页面初始化与 Session State ---
st.set_page_config(page_title="EV RL Testbed", layout="wide")
st.title("🔋 London EV Multi-Agent Evaluation Platform")

if 'all_tracks' not in st.session_state: st.session_state.all_tracks = None
if 'all_stats' not in st.session_state: st.session_state.all_stats = None
if 'last_env' not in st.session_state: st.session_state.last_env = None

# --- 5. 侧边栏 ---
with st.sidebar:
    st.header("📂 Model Repository")
    model_list = [f for f in os.listdir(MODEL_FOLDER) if f.endswith('.pth')]
    selected_model = st.selectbox("Select Model:", options=model_list) if model_list else None
    
    st.divider()
    st.header("⚙️ Settings")
    num_agents = st.slider("Number of Agents", 1, 20, 10)
    soc_limit = st.slider("Charging Threshold (SoC %)", 5.0, 50.0, 25.0)
    expert_w = st.slider("Expert Guidance Weight", 0.0, 1.0, 0.1)
    test_episodes = st.number_input("Episodes", min_value=1, max_value=100, value=5)
    
    run_button = st.button("🚀 Run Evaluation", use_container_width=True)
    
    st.divider()
    # 回放滑块：只有在有了轨迹数据后才生效
    max_ep = len(st.session_state.all_tracks) if st.session_state.all_tracks else 1
    replay_ep = st.slider("⏪ View Episode Trace", 1, max_ep, 1)

# --- 6. 资源加载 ---
@st.cache_resource
def load_eval_assets(_num_agents, model_name):
    env = HexTrafficEnv(num_agents=_num_agents)
    agent = ExpertDQN(state_dim=20, action_dim=6)
    if model_name:
        path = os.path.join(MODEL_FOLDER, model_name)
        agent.policy_net.load_state_dict(torch.load(path, map_location='cpu'))
    return env, agent

if not selected_model: st.stop()
env, agent = load_eval_assets(num_agents, selected_model)
env.soc_threshold = soc_limit

# --- 7. 运行模拟逻辑 ---
if run_button:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_episode_tracks = [] # 记录每一局
    stats = {"success": 0, "out_of_battery": 0, "out_of_road": 0, "timeout": 0, "final_soc": [], "total_reward": 0}

    for ep in range(int(test_episodes)):
        status_text.text(f"Processing Episode {ep+1}/{test_episodes}...")
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            actions = []
            for i in range(num_agents):
                act = agent.select_action(obs[i], env, agent_id=i, training=True, expert_weight=expert_w)
                actions.append(int(act))
            
            obs, rewards, all_done, truncated, infos = env.step(list(actions))
            ep_reward += sum(rewards) / num_agents
            done = all_done or truncated
        
        # 【关键】深度拷贝这一局的轨迹到列表
        all_episode_tracks.append(copy.deepcopy(env.trajectories))
        
        for v in env.vehicles:
            if v.goal: stats["success"] += 1; stats["final_soc"].append(v.soc)
            elif v.finish_status == "out_of_battery": stats["out_of_battery"] += 1
            elif v.finish_status == "out_of_road": stats["out_of_road"] += 1
            else: stats["timeout"] += 1
        
        stats["total_reward"] += ep_reward
        progress_bar.progress((ep + 1) / test_episodes)

    # 存入 Session State 供滑块使用
    st.session_state.all_tracks = all_episode_tracks
    st.session_state.all_stats = stats
    st.session_state.last_env = env
    status_text.success(f"✅ Simulation complete!")

# --- 8. 结果展示 ---
if st.session_state.all_stats:
    s = st.session_state.all_stats
    total = test_episodes * num_agents
    sr = (s["success"] / total) * 100
    
    m1, m2, m3 = st.columns(3)
    m1.metric("ASR", f"{sr:.2f}%")
    m2.metric("Avg SoC", f"{np.mean(s['final_soc']):.2f}%" if s['final_soc'] else "0%")
    m3.metric("Avg Reward", f"{s['total_reward']/test_episodes:.2f}")

    st.divider()
    
    # 根据滑块索引渲染地图
    target_idx = replay_ep - 1
    draw_replay_map(
        st.session_state.last_env, 
        st.session_state.all_tracks[target_idx],
        st.session_state.last_env.agent_positions, # 这里展示终点位置
        selected_model, 
        replay_ep
    )

    st.divider()
    st.subheader("📊 Performance Breakdown")
    st.bar_chart(pd.DataFrame({
        "Outcome": ["Success", "No Battery", "Timeout", "Crashed"],
        "Count": [s["success"], s["out_of_battery"], s["timeout"], s["out_of_road"]]
    }).set_index("Outcome"))
else:
    st.info("👈 Please set parameters and run evaluation to see traces.")