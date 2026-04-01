import streamlit as st
import pandas as pd
import torch
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import h3.api.basic_str as h3_api
import contextily as cx
from shapely.geometry import Polygon, Point
import geopandas as gpd

# 导入你的自定义模块
from mutilEnv import HexTrafficEnv  
from mutilDqfsAgent import ExpertDQN 

# --- 1. 路径与坐标配置 (Path & H3 Config) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 确保文件名与你 GitHub 上的完全一致
MODEL_FILENAME = "refined_final_model.pth" 
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)

# 坐标转换参数（必须与你的 map_processor 一致）
ANCHOR_LAT, ANCHOR_LON = 42.995486, -81.253178
H3_RES = 9
ANCHOR_CELL = h3_api.latlng_to_cell(ANCHOR_LAT, ANCHOR_LON, H3_RES)

# --- 2. 坐标转换工具函数 ---
def get_h3_polygon(i, j):
    """将 IJ 相对坐标转回 Shapely Polygon"""
    try:
        anchor_ij = h3_api.cell_to_local_ij(ANCHOR_CELL, ANCHOR_CELL)
        target_i = i + anchor_ij[0]
        target_j = j + anchor_ij[1]
        target_cell = h3_api.local_ij_to_cell(ANCHOR_CELL, target_i, target_j)
        boundary = h3_api.cell_to_boundary(target_cell)
        # H3 返回 (lat, lon)，Shapely 需要 (lon, lat)
        return Polygon([(lon, lat) for lat, lon in boundary])
    except:
        return None

# --- 3. 地图绘制函数 (基于你的 Matplotlib 逻辑) ---
def draw_static_map(env):
    st.subheader("🗺️ Infrastructure & Agent Monitor (London, ON)")
    
    with st.spinner("Generating map with basemap..."):
        # A. 转换路网 (Roads)
        road_polys = [get_h3_polygon(ij[0], ij[1]) for ij in env.london_main_roads]
        gdf_road = gpd.GeoDataFrame(geometry=[p for p in road_polys if p], crs="EPSG:4326")
        
        # B. 转换充电站 (Chargers)
        l2_list = [get_h3_polygon(ij[0], ij[1]) for ij, lv in env.charging_stations.items() if lv == 'L2']
        l3_list = [get_h3_polygon(ij[0], ij[1]) for ij, lv in env.charging_stations.items() if lv == 'L3']
        
        # C. 转换 Agent 实时位置 (Current Agents)
        agent_points = []
        for pos in env.agent_positions:
            p = get_h3_polygon(pos[0], pos[1])
            if p: agent_points.append(p.centroid)
        gdf_agents = gpd.GeoDataFrame(geometry=agent_points, crs="EPSG:4326")

        # D. 开始绘图
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 统一转为 Web Mercator 投影 (EPSG:3857) 以适配底图
        if not gdf_road.empty:
            gdf_road.to_crs(epsg=3857).plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=0.5, alpha=0.1)
        
        if l2_list:
            gpd.GeoDataFrame(geometry=[p.centroid for p in l2_list], crs="EPSG:4326").to_crs(epsg=3857).plot(
                ax=ax, color='cyan', marker='+', markersize=60, label='L2 Charger'
            )
        if l3_list:
            gpd.GeoDataFrame(geometry=[p.centroid for p in l3_list], crs="EPSG:4326").to_crs(epsg=3857).plot(
                ax=ax, color='red', marker='P', markersize=100, label='L3 Fast Charger'
            )
        
        if not gdf_agents.empty:
            gdf_agents.to_crs(epsg=3857).plot(ax=ax, color='yellow', marker='o', markersize=40, edgecolor='black', label='EV Agents')

        # 添加底图 (CartoDB Positron 风格)
        try:
            cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
        except Exception as e:
            st.error(f"Basemap load failed: {e}")

        ax.set_axis_off()
        plt.legend(loc='upper right', fontsize='small')
        
        # 在 Streamlit 中渲染 Matplotlib 图表
        st.pyplot(fig)

# --- 4. 页面基础设置 ---
st.set_page_config(page_title="EV RL Simulator", layout="wide", page_icon="🔋")
st.title("🔋 London EV Charging Scheduling Platform")

# --- 5. 资源加载 (带缓存) ---
@st.cache_resource
def load_assets(_num_agents):
    env = HexTrafficEnv(num_agents=_num_agents)
    # 确保参数名 state_dim/action_dim 与你的类定义一致
    agent = ExpertDQN(state_dim=20, action_dim=6) 
    if os.path.exists(MODEL_PATH):
        try:
            agent.policy_net.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
            st.sidebar.success(f"✅ Loaded: {MODEL_FILENAME}")
        except:
            st.sidebar.error("❌ Failed to load model weights.")
    return env, agent

# --- 6. 侧边栏交互 ---
with st.sidebar:
    st.header("⚙️ Configuration")
    num_agents = st.slider("Number of Agents", 1, 20, 10)
    soc_limit = st.slider("Charging Threshold (SoC %)", 5.0, 50.0, 25.0)
    expert_w = st.slider("Expert Guidance Weight", 0.0, 1.0, 0.1)
    weather_val = st.select_slider("Weather Factor", options=[0.8, 1.0, 1.2, 1.5, 2.0], value=1.2)
    test_episodes = st.number_input("Episodes", min_value=1, max_value=100, value=10)
    st.divider()
    run_button = st.button("🚀 Run Evaluation", use_container_width=True)

# 初始化环境与模型
env, agent = load_assets(num_agents)
env.soc_threshold = soc_limit
env.weather_factor = weather_val

# --- 7. 运行模拟逻辑 ---
if run_button:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    stats = {
        "success": 0, "out_of_battery": 0, "out_of_road": 0, "timeout": 0,
        "final_soc": [], "total_reward": 0
    }

    for ep in range(int(test_episodes)):
        status_text.text(f"Simulating Episode {ep+1}/{test_episodes}...")
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            actions = []
            for i in range(num_agents):
                # 评估模式：epsilon=0，结合滑块选定的专家权重
                agent.epsilon = 0.0 
                act = agent.select_action(obs[i], env, agent_id=i, training=True, expert_weight=expert_w)
                actions.append(act)
            
            obs, rewards, all_done, truncated, infos = env.step(actions)
            ep_reward += sum(rewards) / num_agents
            done = all_done or truncated
        
        # 统计结果 (对齐你的统计逻辑)
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

    # --- 8. 结果指标看板 ---
    total_samples = test_episodes * num_agents
    success_rate = (stats["success"] / total_samples) * 100
    avg_soc = np.mean(stats["final_soc"]) if stats["final_soc"] else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Individual Success Rate", f"{success_rate:.2f}%", delta=f"{stats['success']}/{total_samples}")
    c2.metric("Avg Final SoC", f"{avg_soc:.2f}%")
    c3.metric("Avg Episode Reward", f"{stats['total_reward']/test_episodes:.2f}")

    st.divider()

    # --- 9. 地图展示 ---
    draw_static_map(env)

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
    st.info("👈 Please set parameters in the sidebar and click 'Run Evaluation'.")