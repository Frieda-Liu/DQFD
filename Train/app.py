import streamlit as st
import pandas as pd
import torch
import numpy as np
import os
import copy
import matplotlib.pyplot as plt
import h3.api.basic_str as h3_api
import contextily as cx
from shapely.geometry import Polygon, Point, LineString
import geopandas as gpd

from mutilEnv import HexTrafficEnv  
from mutilDqfsAgent import ExpertDQN 

# --- Base Config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_FOLDER, exist_ok=True)

ANCHOR_LAT, ANCHOR_LON = 42.995486, -81.253178
H3_RES = 9
ANCHOR_CELL = h3_api.latlng_to_cell(ANCHOR_LAT, ANCHOR_LON, H3_RES)

# --- H3 Utils ---
def get_h3_polygon(i, j):
    try:
        anchor_ij = h3_api.cell_to_local_ij(ANCHOR_CELL, ANCHOR_CELL)
        target_cell = h3_api.local_ij_to_cell(
            ANCHOR_CELL, i + anchor_ij[0], j + anchor_ij[1]
        )
        boundary = h3_api.cell_to_boundary(target_cell)
        return Polygon([(lon, lat) for lat, lon in boundary])
    except:
        return None

def get_latlon_point(i, j):
    try:
        anchor_ij = h3_api.cell_to_local_ij(ANCHOR_CELL, ANCHOR_CELL)
        target_cell = h3_api.local_ij_to_cell(
            ANCHOR_CELL, i + anchor_ij[0], j + anchor_ij[1]
        )
        lat, lon = h3_api.cell_to_latlng(target_cell)
        return lon, lat
    except:
        return None

# --- Draw Map ---
def draw_replay_map(env, tracks, model_name, ep_idx):
    fig, ax = plt.subplots(figsize=(8, 8))

    # roads
    road_polys = [get_h3_polygon(i, j) for i, j in env.london_main_roads]
    gdf_road = gpd.GeoDataFrame(
        geometry=[p for p in road_polys if p], crs="EPSG:4326"
    )
    if not gdf_road.empty:
        gdf_road.to_crs(epsg=3857).plot(
            ax=ax, facecolor='none', edgecolor='blue', linewidth=0.1, alpha=0.05
        )

    # chargers
    l2 = [get_h3_polygon(i, j) for (i, j), lv in env.charging_stations.items() if lv == 'L2']
    l3 = [get_h3_polygon(i, j) for (i, j), lv in env.charging_stations.items() if lv == 'L3']

    if l2:
        gpd.GeoDataFrame(geometry=[p.centroid for p in l2], crs="EPSG:4326")\
            .to_crs(epsg=3857).plot(ax=ax, color='cyan', marker='+', markersize=50, label='L2 Charger')

    if l3:
        gpd.GeoDataFrame(geometry=[p.centroid for p in l3], crs="EPSG:4326")\
            .to_crs(epsg=3857).plot(ax=ax, color='red', marker='P', markersize=80, label='L3 Fast Charger')

    # agents
    for i, track in enumerate(tracks):
        pts = []
        for p in track:
            pt = get_latlon_point(p[0], p[1])
            if pt:
                pts.append(pt)

        if len(pts) < 2:
            continue

        soc = env.vehicles[i].soc if i < len(env.vehicles) else 0.5
        color = (1 - soc, soc, 0)  # red → green

        # path
        gpd.GeoDataFrame(geometry=[LineString(pts)], crs="EPSG:4326")\
            .to_crs(epsg=3857).plot(ax=ax, color=color, linewidth=2)

        # start
        gpd.GeoDataFrame(geometry=[Point(pts[0])], crs="EPSG:4326")\
            .to_crs(epsg=3857).plot(ax=ax, color=color, marker='o', markersize=30)

        # end
        gpd.GeoDataFrame(geometry=[Point(pts[-1])], crs="EPSG:4326")\
            .to_crs(epsg=3857).plot(ax=ax, color=color, marker='*', markersize=100)

    try:
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
    except:
        pass

    ax.set_title(f"Episode {ep_idx} - {model_name}")
    ax.legend(loc='upper right')
    ax.axis('off')

    return fig

# --- UI ---
st.set_page_config(page_title="EV RL Platform", layout="wide")
st.title("🔋 London EV Multi-Agent Charging Platform")

# session state
if 'tracks' not in st.session_state:
    st.session_state.tracks = None
if 'rewards' not in st.session_state:
    st.session_state.rewards = None
if 'env' not in st.session_state:
    st.session_state.env = None

# Sidebar
with st.sidebar:
    st.header("Model Selection")
    models = [f for f in os.listdir(MODEL_FOLDER) if f.endswith(".pth")]
    model_name = st.selectbox("Select Model", models) if models else None

    st.header("Simulation Settings")
    num_agents = st.slider("Number of Agents", 1, 20, 10)
    soc_limit = st.slider("SoC Threshold (%)", 5.0, 50.0, 25.0)
    expert_w = st.slider("Expert Weight", 0.0, 1.0, 0.1)
    episodes = st.number_input("Episodes", 1, 50, 5)

    run = st.button("🚀 Run Simulation", use_container_width=True)

# Load
@st.cache_resource
def load_env(n, model):
    env = HexTrafficEnv(num_agents=n)
    agent = ExpertDQN(20, 6)
    if model:
        path = os.path.join(MODEL_FOLDER, model)
        agent.policy_net.load_state_dict(torch.load(path, map_location='cpu'))
    return env, agent

if not model_name:
    st.info("Please select a model.")
    st.stop()

env, agent = load_env(num_agents, model_name)
env.soc_threshold = soc_limit

# Run Simulation
if run:
    progress = st.progress(0)
    tracks = []
    rewards_all = []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0

        while not done:
            acts = [
                int(agent.select_action(o, env, i, True, expert_w))
                for i, o in enumerate(obs)
            ]
            obs, rewards, all_done, truncated, _ = env.step(acts)
            ep_reward += sum(rewards) / num_agents
            done = all_done or truncated

        rewards_all.append(ep_reward)
        tracks.append(copy.deepcopy(env.trajectories))
        progress.progress((ep + 1) / episodes)

    st.session_state.tracks = tracks
    st.session_state.rewards = rewards_all
    st.session_state.env = env

    st.success("✅ Simulation completed!")

# --- Display ---
if st.session_state.tracks:
    ep_id = st.slider("Select Episode", 1, len(st.session_state.tracks), 1)

    st.caption("Color indicates SoC level: red = low battery, green = high battery")

    with st.spinner("Rendering map..."):
        fig = draw_replay_map(
            st.session_state.env,
            st.session_state.tracks[ep_id - 1],
            model_name,
            ep_id
        )
        st.pyplot(fig)

    st.divider()

    # Reward trend
    st.subheader("📈 Reward Trend")
    st.line_chart(st.session_state.rewards)

    # Metrics
    rewards = np.array(st.session_state.rewards)

    col1, col2, col3 = st.columns(3)
    col1.metric("Average Reward", f"{rewards.mean():.2f}")
    col2.metric("Best Episode", f"{rewards.max():.2f}")
    col3.metric("Worst Episode", f"{rewards.min():.2f}")
