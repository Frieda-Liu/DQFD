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

# --- 3. 绘图函数 ---
def draw_replay_map(env, tracks_to_draw, model_name, ep_idx):
    st.subheader(f"📍 Episode {ep_idx} Trajectory Replay ({model_name})")
    
    with st.spinner(f"Drawing Episode {ep_idx}..."):
        road_polys = [get_h3_polygon(ij[0], ij[1]) for ij in env.london_main_roads]
        gdf_road = gpd.GeoDataFrame(geometry=[p for p in road_polys if p], crs="EPSG:4326")
        l2_list = [get_h3_polygon(ij[0], ij[1]) for ij, lv in env.charging_stations.items() if lv == 'L2']
        l3_list = [get_h3_polygon(ij[0], ij[1]) for ij, lv in env.charging_stations.items() if lv == 'L3']
        
        num_agents = len(tracks_to_draw)
        colors = plt.cm.rainbow(np.linspace(0, 1, num_agents))
        fig, ax = plt.subplots(figsize=(10, 10))
        
        if not gdf_road.empty:
            gdf_road.to_crs(epsg=3857).plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=0.1, alpha=0.05)
        
        if l2_list:
            gpd.GeoDataFrame(geometry=[p.centroid for p in l2_list], crs="EPSG:4326").to_crs(epsg=3857).plot(
                ax=ax, color='cyan', marker='+', markersize=40, label='L2 Charger', alpha=0.4)
        if l3_list:
            gpd.GeoDataFrame(geometry=[p.centroid for p in l3_list], crs="EPSG:4326").to_crs(epsg=3857).plot(
                ax=ax, color='red', marker='P', markersize=80, label='L3 Fast Charger', alpha=0.6)

        for i in range(num_agents):
            c = colors[i]
            track = tracks_to_draw[i]
            pts = [get_latlon_point(p[0], p[1]) for p in track]
            valid_pts = [p for p in pts if p is not None]
            
            if len(valid_pts) >= 2:
                gpd.GeoDataFrame(geometry=[LineString(valid_pts)], crs="EPSG:4326").to_crs(epsg=3857).plot(
                    ax=ax, color=c, linewidth=2, alpha=0.7)
                gpd.GeoDataFrame(geometry=[Point(valid_pts[0])], crs="EPSG:4326").to_crs(epsg=3857).plot(
                    ax=ax, color=c, marker='o', markersize=20, edgecolor='white')
                gpd.GeoDataFrame(geometry=[Point(valid_pts[-1])], crs="EPSG:4326").to_crs(epsg=3857).plot(
                    ax=ax, color=c, marker='*', markersize=100, edgecolor='black', zorder=5)

        try:
            cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
        except: pass
        
        ax.set_axis_off()
        st.pyplot(fig)

# --- 4. 初始化 ---
st.set_page_config(page_title="EV RL Platform", layout="wide", page_icon="🔋")
st.title("🔋 London EV Multi-Agent Evaluation Platform")

if 'all_tracks' not in st.session_state: st.session_state.all_tracks = None
if 'all_stats' not in st.session_state: st.session_state.all_stats = None
if 'last_env_obj' not in st.session_state: st.session_state.last_env_obj = None

# --- 5. 侧边栏 (带有选择器的位置优化) ---
with st.sidebar:
    st.header("📂 模型库")
    model_list = [f for f in os.listdir(MODEL_FOLDER) if f.endswith('.pth')]
    selected_model = st.selectbox("选择模型版本:", options=model_list) if model_list else None
    
    st.divider()
    st.header("⚙️ 仿真设置")
    num_agents = st.slider("智能体数量", 1, 20, 10)
    soc_limit = st.slider("充电决策阈值 (SoC %)", 5.0, 50.0, 25.0)
    expert_w = st.slider("专家引导权重", 0.0, 1.0, 0.1)
    test_episodes = st.number_input("测试轮数 (Episodes)", 1, 100, 5)
    
    # 核心按钮
    run_button = st.button("🚀 开始评估全部 Episode", use_container_width=True)
    
    st.divider()
    
    # --- 这里是你要的“选择第几个”滑块 ---
    st.header("⏪ 轨迹回放控制")
    if st.session_state.all_tracks is not None:
        total_eps = len(st.session_state.all_tracks)
        # 如果只有一局，滑块自动失效；多局时可以滑动
        replay_ep = st.slider(
            "选择要查看的第几个 Episode:", 
            min_value=1, 
            max_value=total_eps, 
            value=1,
            help="滑动以查看不同 Episode 的 Agent 路径"
        )
    else:
        st.info("运行完评估后，这里会出现回放滑块。")
        replay_ep = 1

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

# --- 7. 运行逻辑 ---
if run_button:
    progress_bar = st.progress(0)
    temp_tracks = [] 
    stats = {"success": 0, "out_of_battery": 0, "out_of_road": 0, "timeout": 0, "final_soc": [], "total_reward": 0}

    for ep in range(int(test_episodes)):
        obs, _ = env.reset()
        done = False
        while not done:
            actions = [int(agent.select_action(o, env, agent_id=i, training=True, expert_weight=expert_w)) for i, o in enumerate(obs)]
            obs, rewards, all_done, truncated, infos = env.step(actions)
            done = all_done or truncated
        
        # 存入轨迹
        temp_tracks.append(copy.deepcopy(env.trajectories))
        
        # 统计
        for v in env.vehicles:
            if v.goal: stats["success"] += 1; stats["final_soc"].append(v.soc)
            elif v.finish_status == "out_of_battery": stats["out_of_battery"] += 1
            elif v.finish_status == "out_of_road": stats["out_of_road"] += 1
            else: stats["timeout"] += 1
        
        stats["total_reward"] += sum(rewards) / num_agents
        progress_bar.progress((ep + 1) / test_episodes)

    # 存入缓存
    st.session_state.all_tracks = temp_tracks
    st.session_state.all_stats = stats
    st.session_state.last_env_obj = env
    st.success("✅ 全部评估完成！现在你可以在左侧滑动选择回放了。")

# --- 8. 结果看板与地图渲染 ---
if st.session_state.all_stats:
    s = st.session_state.all_stats
    total = test_episodes * num_agents
    
    m1, m2, m3 = st.columns(3)
    m1.metric("成功率 (ASR)", f"{(s['success']/total)*100:.2f}%")
    m2.metric("平均剩余 SoC", f"{np.mean(s['final_soc']):.2f}%" if s['final_soc'] else "0%")
    m3.metric("累计 Reward", f"{s['total_reward']/test_episodes:.2f}")

    st.divider()
    
    # 渲染地图：使用滑块选中的那一局数据
    draw_replay_map(
        st.session_state.last_env_obj, 
        st.session_state.all_tracks[replay_ep - 1], 
        selected_model, 
        replay_ep
    )