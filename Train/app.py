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

# --- 1. Global Configurations ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_FOLDER, exist_ok=True)

# London Geographic Anchor
ANCHOR_LAT, ANCHOR_LON = 42.995486, -81.253178
H3_RES = 9
ANCHOR_CELL = h3_api.latlng_to_cell(ANCHOR_LAT, ANCHOR_LON, H3_RES)

from mutilEnv import HexTrafficEnv  
from mutilDqfsAgent import ExpertDQN 

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

# --- 3. Renderer ---
def draw_replay_map(env, tracks, model_name, ep_idx):
    fig, ax = plt.subplots(figsize=(10, 10))
    # Road Network
    road_polys = [get_h3_polygon(i, j) for i, j in env.london_main_roads]
    gdf_road = gpd.GeoDataFrame(geometry=[p for p in road_polys if p], crs="EPSG:4326")
    if not gdf_road.empty:
        gdf_road.to_crs(epsg=3857).plot(ax=ax, facecolor='none', edgecolor='#3182bd', linewidth=0.1, alpha=0.1)

    num_to_draw = len(tracks)
    for i in range(num_to_draw):
        track = tracks[i]
        pts = [get_latlon_point(p[0], p[1]) for p in track]
        pts = [p for p in pts if p is not None]
        if len(pts) < 2: continue

        soc = 100.0
        if i < len(env.vehicles): soc = env.vehicles[i].soc
        s = np.clip(soc / 100.0, 0, 1)
        agent_color = mcolors.to_hex((1 - s, s, 0.2))

        gpd.GeoDataFrame(geometry=[LineString(pts)], crs="EPSG:4326").to_crs(epsg=3857).plot(
            ax=ax, color=agent_color, linewidth=2.5, alpha=0.8, 
            label=f"Agent {i} ({int(soc)}% SoC)" if num_to_draw <= 10 else ""
        )
        gpd.GeoDataFrame(geometry=[Point(pts[0])], crs="EPSG:4326").to_crs(epsg=3857).plot(
            ax=ax, color=agent_color, marker='o', markersize=30, edgecolor='white')
        gpd.GeoDataFrame(geometry=[Point(pts[-1])], crs="EPSG:4326").to_crs(epsg=3857).plot(
            ax=ax, color=agent_color, marker='*', markersize=150, edgecolor='black', zorder=10)

    try: cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
    except: pass
    ax.set_title(f"Episode {ep_idx} Analysis | Model: {model_name}", fontsize=14)
    if num_to_draw <= 10: ax.legend(loc='upper right', fontsize='x-small', ncol=2)
    ax.set_axis_off()
    return fig

# --- 4. Initialization ---
st.set_page_config(page_title="EV RL Platform", layout="wide", page_icon="🔋")
st.title("🔋 London EV Routing & Evaluation Platform")

if 'all_tracks' not in st.session_state: st.session_state.all_tracks = None
if 'all_stats' not in st.session_state: st.session_state.all_stats = None
if 'final_env' not in st.session_state: st.session_state.final_env = None

# --- 5. Sidebar ---
with st.sidebar:
    st.header("📂 Model Repository")
    model_list = sorted([f for f in os.listdir(MODEL_FOLDER) if f.endswith('.pth')])
    selected_model = st.selectbox("Select Model:", options=model_list) if model_list else None
    
    st.divider()
    st.header("⚙️ Simulation Settings")
    num_agents = st.slider("Number of Agents", 1, 20, 10)
    soc_limit = st.slider("Charging Threshold %", 5.0, 50.0, 25.0)
    weather_factor = st.select_slider("Weather Factor (Consumption)", options=[1.0, 1.2, 1.5, 2.0], value=1.0)
    expert_w = st.slider("Expert Guidance Weight", 0.0, 1.0, 0.1)
    test_episodes = st.number_input("Test Episodes", 1, 100, 5)
    
    run_button = st.button("🚀 Run Full Evaluation", use_container_width=True)
    
    st.divider()
    st.header("⏪ Replay Control")
    # SAFE REPLAY LOGIC: Check if tracks exist and handle Single Episode Case
    if st.session_state.all_tracks and len(st.session_state.all_tracks) > 0:
        total_eps = len(st.session_state.all_tracks)
        if total_eps > 1:
            replay_ep = st.slider("Select Episode to Replay:", 1, total_eps, 1)
        else:
            st.info("Single episode available.")
            replay_ep = 1
    else:
        st.info("Run evaluation first.")
        replay_ep = 1

# --- 6. Execution ---
if run_button:
    st.session_state.all_tracks = None
    st.session_state.all_stats = None
    
    env = HexTrafficEnv(num_agents=num_agents)
    env.soc_threshold = soc_limit
    env.weather_factor = weather_factor # Inject weather impact
    
    agent = ExpertDQN(state_dim=20, action_dim=6)
    if selected_model:
        agent.policy_net.load_state_dict(torch.load(os.path.join(MODEL_FOLDER, selected_model), map_location='cpu'))

    progress_bar = st.progress(0)
    status_text = st.empty()
    temp_tracks = []
    
    current_run_total = int(test_episodes * num_agents)
    stats = {"success": 0, "out_of_battery": 0, "out_of_road": 0, "timeout": 0, 
             "final_soc": [], "total_reward": 0, "total_deployed": current_run_total}

    for ep in range(int(test_episodes)):
        status_text.text(f"Processing Episode {ep+1}/{test_episodes}...")
        obs, _ = env.reset()
        done = False
        while not done:
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

    st.session_state.all_tracks = temp_tracks
    st.session_state.all_stats = stats
    st.session_state.final_env = env
    status_text.success(f"✅ Simulation Complete! (Weather: {weather_factor}x consumption)")
    st.rerun()

# --- 7. Result Dashboard ---
if st.session_state.all_stats and st.session_state.all_tracks:
    s = st.session_state.all_stats
    total = s["total_deployed"]
    asr = (s["success"] / total) * 100
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Success Rate (ASR)", f"{asr:.2f}%", f"{s['success']}/{total}")
    c2.metric("Avg Remaining SoC", f"{np.mean(s['final_soc']):.2f}%" if s['final_soc'] else "0%")
    c3.metric("Avg Reward/Episode", f"{s['total_reward']/len(st.session_state.all_tracks):.2f}")

    st.divider()
    
    # Safe indexing for single or multiple episodes
    idx = min(replay_ep - 1, len(st.session_state.all_tracks) - 1)
    fig = draw_replay_map(st.session_state.final_env, st.session_state.all_tracks[idx], selected_model, replay_ep)
    st.pyplot(fig)

    st.divider()
    st.subheader("📊 Performance Statistics Breakdown")
    breakdown_df = pd.DataFrame({
        "Status": ["Success", "Battery Empty", "Timeout", "Crashed"],
        "Count": [s["success"], s["out_of_battery"], s["timeout"], s["out_of_road"]]
    }).set_index("Status")
    st.bar_chart(breakdown_df)
else:
    st.info("👈 Configure settings and click 'Run Full Evaluation' to begin.")