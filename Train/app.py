import streamlit as st
import pandas as pd
import torch
import numpy as np
import os
import time

from mutilEnv import HexTrafficEnv  
from mutilDqfsAgent import ExpertDQN 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILENAME = "mutilDqfd_final_rl_model.pth" 
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)

# --- base web page ---
st.set_page_config(page_title="EV RL Simulator", layout="wide", page_icon="🔋")
st.title("🔋 London EV Charging Scheduling Platform")
st.markdown(f"Current model: `{MODEL_FILENAME}`")

# --- variables ---
with st.sidebar:
    st.header("⚙️ Simulation Parameters")
    num_agents = st.slider("Number of Agents", 1, 20, 10)
    soc_limit = st.slider("Charging Decision Threshold (SoC %)", 5.0, 50.0, 25.0)
    
    st.divider()
    st.header("🧠 Expert Weight")
    
    # --- expert weight slider ---
    expert_w = st.slider(
        "Expert Weight", 
        0.0, 
        1.0, 
        0.2, 
        help="0.0 = no expert, 1.0 = pure expert heuristic"
    )
    
    weather_val = st.select_slider(
        "Weather Factor", 
        options=[0.8, 1.0, 1.2, 1.5, 2.0], 
        value=1.2
    )
    
    test_episodes = st.number_input("Episodes", min_value=1, max_value=50, value=10)
    
    st.divider()
    run_button = st.button("🚀 Start Evaluation", use_container_width=True)

# --- load resources ---
@st.cache_resource
def load_assets(_num_agents):
    env = HexTrafficEnv(num_agents=_num_agents)
    
    agent = ExpertDQN(state_dim=20, action_dim=6) 
    
    if os.path.exists(MODEL_PATH):
        try:
            agent.policy_net.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
            st.sidebar.success("✅ Model loaded")
        except Exception as e:
            st.sidebar.error(f"❌ Failed to load model: {e}")
    else:
        st.sidebar.warning(f"⚠️ Model not found: {MODEL_FILENAME}")
        
    return env, agent

env, agent = load_assets(num_agents)

env.soc_threshold = soc_limit
env.weather_factor = weather_val

# --- start run ---
if run_button:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    stats = {
        "success": 0, "timeout": 0, "no_battery": 0, "crash": 0,
        "final_soc": [], "rewards": []
    }

    for ep in range(int(test_episodes)):
        status_text.text(f"Simulating episode {ep+1}/{test_episodes}...")
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            actions = []
            for i in range(num_agents):
                agent.epsilon = 0.0 
                act = agent.select_action(
                    obs[i], 
                    env, 
                    agent_id=i, 
                    training=True, 
                    expert_weight=expert_w
                )
                actions.append(act)
            
            obs, rewards, all_done, truncated, infos = env.step(actions)
            ep_reward += sum(rewards) / num_agents
            done = all_done or truncated
        
        # record results
        for info in infos:
            res = info.get("reason", "")
            if res == "success": stats["success"] += 1
            elif res == "out_of_battery": stats["no_battery"] += 1
            elif res == "timeout_penalty": stats["timeout"] += 1
            elif res == "out_of_road": stats["crash"] += 1
            if "soc" in info: stats["final_soc"].append(info["soc"])
            
        stats["rewards"].append(ep_reward)
        progress_bar.progress((ep + 1) / test_episodes)

    status_text.success("✅ Finished!")

    # --- results panel ---
    total = test_episodes * num_agents
    sr = (stats["success"] / total) * 100
    avg_soc = np.mean(stats["final_soc"]) if stats["final_soc"] else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Success Rate (SR)", f"{sr:.1f}%")
    col2.metric("Average Remaining SoC", f"{avg_soc:.1f}%")
    col3.metric("Out-of-Battery Events", stats["no_battery"])
    col4.metric("Timeout Events", stats["timeout"])

    st.divider()

    # --- charts ---
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📌 Task Outcome Distribution")
        res_df = pd.DataFrame({
            "Status": ["Success", "No Battery", "Timeout", "Crash"],
            "Count": [stats["success"], stats["no_battery"], stats["timeout"], stats["crash"]]
        })
        st.bar_chart(res_df.set_index("Status"))

    with c2:
        st.subheader("📈 Final SoC Distribution")
        if stats["final_soc"]:
            st.area_chart(pd.DataFrame(stats["final_soc"], columns=["Final SoC"]))
        else:
            st.write("No SoC data available")

    with st.expander("Click to view raw data"):
        st.write(stats)

else:
    st.info("👈 Adjust parameters on the left (including expert weight), then click Run.")
