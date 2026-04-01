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
# Ensure your .pth models are in 'Train/models/'
MODEL_FOLDER = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_FOLDER, exist_ok=True)

# London Geographic Anchor (Consistent with map_processor)
ANCHOR_LAT, ANCHOR_LON = 42.995486, -81.253178
H3_RES = 9
ANCHOR_CELL = h3_api.latlng_to_cell(ANCHOR_LAT, ANCHOR_LON, H3_RES)

# --- 2. H3 Coordinate Utils ---
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

# --- 3. Dynamic Replay Map Renderer ---
def draw_replay_map(env, tracks, model_name, ep_idx):
    """Render a clean trajectory map with SoC-based coloring"""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Background: Road Network
    road_polys = [get_h3_polygon(i, j) for i, j in env.london_main_roads]
    gdf_road = gpd.GeoDataFrame(geometry=[p for p in road_polys if p], crs="EPSG:4326")
    if not gdf_road.empty:
        gdf_road.to_crs(epsg=3857).plot(ax=ax, facecolor='none', edgecolor='#3182bd', linewidth=0.1, alpha=0.1)

    # Infrastructure: Chargers
    l2 = [get_h3_polygon(i, j) for (i, j), lv in env.charging_stations.items() if lv == 'L2']
    l3 = [get_h3_polygon(i, j) for (i, j), lv in env.charging_stations.items() if lv == 'L3']
    if l2:
        gpd.GeoDataFrame(geometry=[p.centroid for p in l2], crs="EPSG:4326").to_crs(epsg=3857).plot(
            ax=ax, color='#74a9cf', marker='+', markersize=60, label='L2 Charger', alpha=0.6)
    if l3:
        gpd.GeoDataFrame(geometry=[p.centroid for p in l3], crs="EPSG:4326").to_crs(epsg=3857).plot(
            ax=ax, color='#e31a1c', marker='P', markersize=100, label='L3 Fast Charger', alpha=0.8)

    # Dynamic: Individual Agent Trajectories
    num_agents = len(tracks)
    for i, track in enumerate(tracks):
        pts = [get_latlon_point(p[0], p[1]) for p in track]
        pts = [p for p in pts if p is not None]
        if len(pts) < 2: continue

        # Logic: Map SoC to Color (Green for High, Red for Low)
        current_soc = 100.0
        if i < len(env.vehicles):
            current_soc = env.vehicles[i].soc
        
        s = np.clip(current_soc / 100.0, 0, 1)
        agent_color = mcolors.to_hex((1 - s, s, 0.2)) # Dynamic Color: Red -> Green

        # Draw Path
        gpd.GeoDataFrame(geometry=[LineString(pts)], crs="EPSG:4326").to_crs(epsg=3857).plot(
            ax=ax, color=agent_color, linewidth=2.5, alpha=0.7, 
            label=f"Agent {i} ({int(current_soc)}% SoC)" if num_agents <= 10 else ""
        )
        # Start (Circle) & End (*)
        gpd.GeoDataFrame(geometry=[Point(pts[0])], crs="EPSG:4326").to_crs(epsg=3857).plot(
            ax=ax, color=agent_color, marker='o', markersize=30, edgecolor='white')
        gpd.GeoDataFrame(geometry=[Point(pts[-1])], crs="EPSG:4326").to_crs(epsg=3857).plot(
            ax=ax, color=agent_color, marker='*', markersize=150, edgecolor='black', zorder=10)

    try:
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
    except: pass

    ax.set_title(f"Episode {ep_idx} Trajectory - Final SoC Status", fontsize=14)
    if num_agents <= 10:
        ax.legend(loc='upper right', fontsize='x-small', ncol=2, framealpha=0.5)
    
    ax.set_axis_off()
    return fig

# --- 4. Page Layout & Session Initialization ---
st.set_page_config(page_title="EV RL Platform", layout="wide", page_icon="🔋")
st.title("🔋 London EV Multi-Agent Routing Platform")

if 'all_tracks' not in st.session_state: st.session_state.all_tracks = None
if 'all_stats' not in st.session_state: st.session_state.all_stats = None
if 'cached_env' not in st.session_state: st.session_state.cached_env = None

# --- 5. Sidebar Controls ---
with st.sidebar:
    st.header("📂 Model Repository")
    model_list = sorted([f for f in os.listdir(MODEL_FOLDER) if f.endswith('.pth')])
    selected_model = st.selectbox("Select Model Version:", options=model_list) if model_list else None
    
    st.divider()
    st.header("⚙️ Simulation Settings")
    num_agents = st.slider("Number of Agents", 1, 20, 10)
    soc_limit = st.slider("Charging SoC Threshold %", 5.0, 50.0, 25.0)
    expert_w = st.slider("Expert Guidance Weight", 0.0, 1.0, 0.1)
    test_episodes = st.number_input("Total Episodes", 1, 100, 5)
    
    run_button = st.button("🚀 Run Full Evaluation", use_container_width=True)
    
    st.divider()
    st.header("⏪ Replay Control")
    # Interactive Slider for Replaying specific episodes
    if st.session_state.all_tracks is not None:
        total_eps = len(st.session_state.all_tracks)
        replay_ep = st.slider("Select Episode to Replay:", 1, total_eps, 1)
    else:
        st.info("Run simulation first to enable replay slider.")
        replay_ep = 1

# --- 6. Asset Loading ---
@st.cache_resource
def load_assets(_num_agents, model_name):
    env = HexTrafficEnv(num_agents=_num_agents)
    agent = ExpertDQN(state_dim=20, action_dim=6)
    if model_name:
        path = os.path.join(MODEL_FOLDER, model_name)
        agent.policy_net.load_state_dict(torch.load(path, map_location='cpu'))
    return env, agent

if not selected_model: st.stop()
env, agent = load_assets(num_agents, selected_model)
env.soc_threshold = soc_limit

# --- 7. Execution Logic ---
if run_button:
    # Clear state for new simulation
    st.session_state.all_tracks = None
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    temp_tracks = [] 
    stats = {"success": 0, "out_of_battery": 0, "out_of_road": 0, "timeout": 0, "final_soc": [], "total_reward": 0}

    for ep in range(int(test_episodes)):
        status_text.text(f"Running Episode {ep+1}/{test_episodes}...")
        obs, _ = env.reset()
        done = False
        while not done:
            # Inference: Actions for all agents
            actions = [int(agent.select_action(o, env, i, training=True, expert_weight=expert_w)) for i, o in enumerate(obs)]
            obs, rewards, all_done, truncated, infos = env.step(actions)
            done = all_done or truncated
        
        # Deep copy to store this episode's trajectory
        temp_tracks.append(copy.deepcopy(env.trajectories))
        
        # Outcome Tracking
        for v in env.vehicles:
            if v.goal: 
                stats["success"] += 1
                stats["final_soc"].append(v.soc)
            elif v.finish_status == "out_of_battery": stats["out_of_battery"] += 1
            elif v.finish_status == "out_of_road": stats["out_of_road"] += 1
            else: stats["timeout"] += 1
        
        stats["total_reward"] += sum(rewards) / num_agents
        progress_bar.progress((ep + 1) / test_episodes)

    # Persist to Session State
    st.session_state.all_tracks = temp_tracks
    st.session_state.all_stats = stats
    st.session_state.cached_env = env
    status_text.success("✅ Full Evaluation Finished!")
    st.rerun() # Refresh UI to show metrics and slider

# --- 8. Result Dashboard ---
if st.session_state.all_stats:
    s = st.session_state.all_stats
    total = test_episodes * num_agents
    asr = (s["success"] / total) * 100
    
    # KPIs
    c1, c2, c3 = st.columns(3)
    c1.metric("Individual Success Rate (ASR)", f"{asr:.2f}%", f"{s['success']}/{total}")
    c2.metric("Avg Remaining SoC", f"{np.mean(s['final_soc']):.2f}%" if s['final_soc'] else "0%")
    c3.metric("Avg Reward/Episode", f"{s['total_reward']/test_episodes:.2f}")

    st.divider()
    
    # Visual Map: Renders selected episode from history
    fig = draw_replay_map(
        st.session_state.cached_env, 
        st.session_state.all_tracks[replay_ep - 1], 
        selected_model, 
        replay_ep
    )
    st.pyplot(fig)

    # Bar Chart for Breakdown
    st.subheader("📊 Performance Statistics Breakdown")
    breakdown_df = pd.DataFrame({
        "Status": ["Success", "Battery Empty", "Timeout", "Crashed"],
        "Count": [s["success"], s["out_of_battery"], s["timeout"], s["out_of_road"]]
    }).set_index("Status")
    st.bar_chart(breakdown_df)
else:
    st.info("👈 Configure simulation settings and click 'Run Full Evaluation' to start.")