import streamlit as st
import pandas as pd
import torch
import numpy as np
import os
import time
import h3.api.basic_str as h3_api
import pydeck as pdk

from mutilEnv import HexTrafficEnv  
from mutilDqfsAgent import ExpertDQN 

# --- 1. 路径与坐标配置 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILENAME = "mutilDqfd_final_rl_model.pth" 
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)

ANCHOR_LAT, ANCHOR_LON = 42.995486, -81.253178
H3_RES = 9
ANCHOR_CELL = h3_api.latlng_to_cell(ANCHOR_LAT, ANCHOR_LON, H3_RES)
ANCHOR_IJ = h3_api.cell_to_local_ij(ANCHOR_CELL, ANCHOR_CELL)

# --- 2. 工具函数 ---
def ij_to_latlon(di, dj):
    real_i = int(di + ANCHOR_IJ[0])
    real_j = int(dj + ANCHOR_IJ[1])
    try:
        cell = h3_api.local_ij_to_cell(ANCHOR_CELL, real_i, real_j)
        lat, lon = h3_api.cell_to_latlng(cell)
        return lat, lon
    except:
        return ANCHOR_LAT, ANCHOR_LON

def draw_pdk_map(env):
    st.subheader("🗺️ Live Map: Vehicle & Infrastructure Monitor")
    agent_list = []
    for i in range(env.num_agents):
        lat, lon = ij_to_latlon(env.agent_positions[i][0], env.agent_positions[i][1])
        color = [0, 255, 100] if env.vehicles[i].goal else [255, 50, 50]
        agent_list.append({
            "name": f"Agent {i}", "lat": lat, "lon": lon,
            "soc": f"{env.vehicles[i].soc:.1f}%", "color": color
        })

    charger_list = []
    for (ci, cj), level in env.charging_stations.items():
        clat, clon = ij_to_latlon(ci, cj)
        charger_list.append({
            "lat": clat, "lon": clon,
            "color": [255, 200, 0, 160] if level == "L3" else [100, 150, 255, 100]
        })

    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/dark-v10",
        initial_view_state=pdk.ViewState(latitude=ANCHOR_LAT, longitude=ANCHOR_LON, zoom=12, pitch=40),
        layers=[
            pdk.Layer("ScatterplotLayer", charger_list, get_position="[lon, lat]", get_color="color", get_radius=100),
            pdk.Layer("ScatterplotLayer", agent_list, get_position="[lon, lat]", get_color="color", get_radius=150, pickable=True),
            pdk.Layer("TextLayer", agent_list, get_position="[lon, lat]", get_text="soc", get_size=15, get_color=[255, 255, 255], get_alignment_baseline="'bottom'")
        ],
        tooltip={"text": "{name}\nSoC: {soc}"}
    ))

# --- 3. 页面设置 ---
st.set_page_config(page_title="EV RL Simulator", layout="wide", page_icon="🔋")
st.title("🔋 London EV Charging Scheduling Platform")

# --- 4. 资源加载 ---
@st.cache_resource
def load_assets(_num_agents):
    env = HexTrafficEnv(num_agents=_num_agents)
    agent = ExpertDQN(state_dim=20, action_dim=6) 
    if os.path.exists(MODEL_PATH):
        try:
            agent.policy_net.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
            st.sidebar.success("✅ Model weights loaded")
        except:
            st.sidebar.error("❌ Weight load failed")
    return env, agent

# --- 5. 侧边栏交互 ---
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

# --- 6. 运行逻辑 ---
if run_button:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 初始化统计数据 (与你提供的逻辑一致)
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
                # 评估模式：结合专家权重
                act = agent.select_action(obs[i], env, agent_id=i, training=True, expert_weight=expert_w)
                actions.append(act)
            
            obs, rewards, all_done, truncated, infos = env.step(actions)
            ep_reward += sum(rewards) / num_agents
            done = all_done or truncated
        
        # --- 核心：对齐你的统计逻辑 ---
        # 每一局结束，遍历所有车辆查看其最终状态
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

    status_text.success("✅ Evaluation Finished!")

    # --- 7. 结果指标展示 ---
    total_samples = test_episodes * num_agents
    success_rate = (stats["success"] / total_samples) * 100
    avg_soc = np.mean(stats["final_soc"]) if stats["final_soc"] else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Individual Success Rate", f"{success_rate:.2f}%", delta=f"{stats['success']}/{total_samples}")
    c2.metric("Avg Final SoC", f"{avg_soc:.2f}%")
    c3.metric("Avg Episode Reward", f"{stats['total_reward']/test_episodes:.2f}")

    st.divider()

    # --- 8. 地图展示 ---
    draw_pdk_map(env)

    st.divider()

    # --- 9. 任务状态分布图 ---
    st.subheader("📌 Task Outcome Distribution (Agent Count)")
    res_df = pd.DataFrame({
        "Status": ["Success", "Battery Empty", "Timeout", "Crashed"],
        "Count": [stats["success"], stats["out_of_battery"], stats["timeout"], stats["out_of_road"]]
    })
    st.bar_chart(res_df.set_index("Status"))

    with st.expander("Show raw statistics"):
        st.write(stats)

else:
    st.info("👈 Please set the configuration in the sidebar and click 'Run Evaluation'.")