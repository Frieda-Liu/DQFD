import streamlit as st
import pandas as pd
import torch
import numpy as np
from mutilEnv import HexTrafficEnv  
from mutilDqfsAgent import DQNAgent 

# --- 页面基础设置 ---
st.set_page_config(page_title="EV RL Simulator", layout="wide")
st.title("🔋 伦敦电动车多智能体补能调度模拟器")

# --- 侧边栏：参数调整 ---
with st.sidebar:
    st.header("实验参数设置")
    num_agents = st.slider("智能体数量 (Agents)", 1, 15, 5)
    soc_limit = st.slider("充电决策阈值 (SoC %)", 10.0, 50.0, 20.0)
    weather = st.selectbox("天气因子", [1.0, 1.2, 1.5, 1.8], index=1)
    
    st.divider()
    run_button = st.button("🚀 开始 100 局评估测试")

# --- 初始化环境与模型 ---
@st.cache_resource # 防止重复加载模型
def load_model():
    env = HexTrafficEnv(num_agents=num_agents)
    # 假设你的 Agent 初始化需要这些参数
    agent = DQNAgent(state_size=20, action_size=6) 
    agent.policy_net.load_state_dict(torch.load("mutilDqfd_final_rl_model.pth", map_location='cpu'))
    return env, agent

env, agent = load_model()
env.soc_threshold = soc_limit # 动态修改环境里的阈值

# --- 运行模拟与展示 ---
if run_button:
    st.info(f"正在使用模型执行 {num_agents} 个 Agent 的补能压力测试...")
    
    # 这里放你之前的 Evaluation 逻辑
    # 记录结果：success_count, avg_soc, timeouts, etc.
    
    # --- 结果展示面板 ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("successful rate", f"{success_rate}%", delta="Phase 7.1")
    with col2:
        st.metric("average soc left", f"{avg_soc}%")
    with col3:
        st.metric("time of out of battery", f"{timeouts} / {no_battery}")

    # --- 可视化图表 ---
    st.subheader("能源消耗分布统计")
    # chart_data = pd.DataFrame(...)
    # st.bar_chart(chart_data)