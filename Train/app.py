import streamlit as st
import pandas as pd
import torch
import numpy as np
import os
import time
import copy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import h3.api.basic_str as h3_api
import contextily as cx
from shapely.geometry import Polygon, Point, LineString
import geopandas as gpd

# Import custom modules
from mutilEnv import HexTrafficEnv  
from mutilDqfsAgent import ExpertDQN 

# --- 1. Global Configurations ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_FOLDER, exist_ok=True)

ANCHOR_LAT, ANCHOR_LON = 42.995486, -81.253178
H3_RES = 9
ANCHOR_CELL = h3_api.latlng_to_cell(ANCHOR_LAT, ANCHOR_LON, H3_RES)

# --- 2. Coordinate Utils ---
def get_h3_polygon(i, j):
    try:
        anchor_ij = h3_api.cell_to_local_ij(ANCHOR_CELL, ANCHOR_CELL)
        target_cell = h3_api.local_ij_to_cell(ANCHOR_CELL, i + anchor_ij[0], j + anchor_ij[1])
        boundary = h3_api.cell_to_boundary(target_cell)
        return Polygon([(lon, lat) for lat, lon in boundary])
    except: return None

def get_latlon_point(i, j):
    try:
        anchor_ij = h3_api.cell_to_local_ij(ANCHOR_CELL, ANCHOR_CELL)
        target_cell = h3_api.local_ij_to_cell(ANCHOR_CELL, i + anchor_ij[0], j + anchor_ij[1])
        lat, lon = h3_api.cell_to_latlng(target_cell)
        return lon, lat
    except: return None

# --- 3. Robust Map Renderer ---
def draw_replay_map(env, tracks, model_name, ep_idx):
    """Render trajectories for ONLY the current agents in the track data"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Roads & Infrastructure
    road_polys = [get_h3_polygon(i, j) for i, j in env.london_main_roads]
    gdf_road = gpd.GeoDataFrame(geometry=[p for p in road_polys if p], crs="EPSG:4326")
    if not gdf_road.empty:
        gdf_road.to_crs(epsg=3857).plot(ax=ax, facecolor='none', edgecolor='#3182bd', linewidth=0.1, alpha=0.1)

    l2 = [get_h3_polygon(i, j) for (i, j), lv in env.charging_stations.items() if lv == 'L2']
    l3 = [get_h3_polygon(i, j) for (i, j), lv in env.charging_stations.items() if lv == 'L3']
    if l2:
        gpd.GeoDataFrame(geometry=[p.centroid for p in l2], crs="EPSG:4326").to_crs(epsg=3857).plot(
            ax=ax, color='#74a9cf', marker='+', markersize=60, label='L2 Charger', alpha=0.6)
    if l3:
        gpd.GeoDataFrame(geometry=[p.centroid for p in l3], crs="EPSG:4326").to_crs(epsg=3857).plot(
            ax=ax, color='#e31a1c', marker='P', markersize=100, label='L3 Fast Charger', alpha=0.8)

    # DRAW ONLY ACTIVE AGENTS
    # tracks is a list of trajectories. len(tracks) MUST equal num_agents from the run.
    num_current_agents = len(tracks)
    
    for i in range(num_current_agents):
        track = tracks[i]
        pts = [get_latlon_point(p[0], p[1]) for p in track]
        pts = [p for p in pts if p is not None]
        if len(pts) < 2: continue

        # Visual SoC feedback
        soc = 100.0
        if i < len(env.vehicles): soc = env.vehicles[i].soc
        s = np.clip(soc / 100.0, 0, 1)
        agent_color = mcolors.to_hex((1 - s, s, 0.2))

        gpd.GeoDataFrame(geometry=[LineString(pts)], crs="EPSG:4326").to_crs(epsg=3857).plot(
            ax=ax, color=agent_color, linewidth=2.5, alpha=0.8, 
            label=f"Agent {i} ({int(soc)}% SoC)" if num_current_agents <= 10 else ""
        )
        gpd.GeoDataFrame(geometry=[Point(pts[0])], crs="EPSG:4326").to_crs(epsg=3857).plot(
            ax=ax, color=agent_color, marker='o', markersize=30, edgecolor='white')
        gpd.GeoDataFrame(geometry=[Point(pts[-1])], crs="EPSG:4326").to_crs(epsg=3857).plot(
            ax=ax, color=agent_color, marker='*', markersize=150, edgecolor='black', zorder=10)

    try:
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
    except: pass

    ax.set_title(f"Episode {ep_idx} Analysis | Active Agents: {num_current_agents}", fontsize=14)
    if num_current_agents <= 10:
        ax.legend(loc='upper right', fontsize='x-small', ncol=2, framealpha=0.5)
    ax.set_axis_off()
    return fig

# --- 4. Initialization ---
st.set_page_config(page_title="EV RL Platform", layout="wide", page_icon="🔋")
st.title("🔋 London EV Evaluation Platform")

if 'all_tracks' not in st.session_state: st.session_state.all_tracks = None
if 'all_stats' not in st.session_state: st.session_state.all_stats = None
if 'active_env' not in st.session_state: st.session_state.active_env = None

# --- 5. Sidebar ---
with st.sidebar:
    # 🧪 Debug Tool: Force Clear Cache if things get stuck
    if st.button("🧹 Emergency Cache Clear"):
        st.cache_resource.clear()
        st.session_state.all_tracks = None
        st.session_state.all_stats = None
        st.rerun()

    st.header("📂 Models")
    model_list = sorted([f for f in os.listdir(MODEL_FOLDER) if f.endswith('.pth')])
    selected_model = st.selectbox("Model:", options=model_list) if model_list else None
    
    st.divider()
    st.header("⚙️ Settings")
    num_agents = st.slider("Number of Agents", 1, 20, 10)
    soc_limit = st.slider("Charging SoC Threshold %", 5.0, 50.0, 25.0)
    expert_w = st.slider("Expert Weight", 0.0, 1.0, 0.1)
    test_episodes = st.number_input("Episodes", 1, 100, 5)
    
    run_button = st.button("🚀 Run Full Evaluation", use_container_width=True)
    
    st.divider()
    if st.session_state.all_tracks:
        replay_ep = st.slider("View Episode:", 1, len(st.session_state.all_tracks), 1)
    else:
        replay_ep = 1

# --- 6. Manual Env Loading (No Caching for Env to prevent Sticky Agents) ---
def get_fresh_assets(n_agents, model_name):
    # This creates a brand new env every time you call it
    env = HexTrafficEnv(num_agents=n_agents)
    agent = ExpertDQN(state_dim=20, action_dim=6)
    if model_name:
        path = os.path.join(MODEL_FOLDER, model_name)
        agent.policy_net.load_state_dict(torch.load(path, map_location='cpu'))
    return env, agent

if not selected_model: st.stop()

# --- 7. Execution ---
if run_button:
    # 1. Physically clear old data
    st.session_state.all_tracks = None
    st.session_state.all_stats = None
    
    # 2. Get a fresh environment based on the CURRENT slider value
    env, agent = get_fresh_assets(num_agents, selected_model)
    env.soc_threshold = soc_limit
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    temp_tracks = []
    
    # Track exactly how many agents we are deploying
    current_total_deployed = int(test_episodes * num_agents)
    stats = {"success": 0, "out_of_battery": 0, "out_of_road": 0, "timeout": 0, 
             "final_soc": [], "total_reward": 0, "n_run": current_total_deployed}

    for ep in range(int(test_episodes)):
        status_text.text(f"Running Episode {ep+1}...")
        obs, _ = env.reset()
        done = False
        while not done:
            # Multi-agent action loop
            actions = [int(agent.select_action(o, env, i, True, expert_w)) for i, o in enumerate(obs)]
            obs, rewards, all_done, truncated, _ = env.step(actions)
            done = all_done or truncated
        
        temp_tracks.append(copy.deepcopy(env.trajectories))
        for v in env.vehicles:
            if v.goal: 
                stats["success"] += 1
                stats["final_soc"].append(v.soc)
            elif v.finish_status == "out_of_battery": stats["out_of_battery"] += 1
            elif v.finish_status == "out_of_road": stats["out_of_road"] += 1
            else: stats["timeout"] += 1
        stats["total_reward"] += sum(rewards) / num_agents
        progress_bar.progress((ep + 1) / test_episodes)

    # Save to state
    st.session_state.all_tracks = temp_tracks
    st.session_state.all_stats = stats
    st.session_state.active_env = env
    status_text.success("✅ Simulation Finished!")
    st.rerun()

# --- 8. Dashboard ---
if st.session_state.all_stats:
    s = st.session_state.all_stats
    total = s["n_run"] # Use the locked number
    asr = (s["success"] / total) * 100
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Success Rate (ASR)", f"{asr:.2f}%", f"{s['success']}/{total}")
    c2.metric("Avg Final SoC", f"{np.mean(s['final_soc']):.2f}%" if s['final_soc'] else "0%")
    c3.metric("Avg Reward", f"{s['total_reward']/test_episodes:.2f}")

    st.divider()
    # Draw map using the tracks from the RELEVANT episode
    fig = draw_replay_map(st.session_state.active_env, st.session_state.all_tracks[replay_ep-1], selected_model, replay_ep)
    st.pyplot(fig)
else:
    st.info("👈 Select settings and click 'Run Full Evaluation'.")