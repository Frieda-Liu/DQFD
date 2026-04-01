import streamlit as st
import pandas as pd
import torch
import numpy as np
import os
import time
import matplotlib.pyplot as plt

# 导入你自己的模块
from mutilEnv import HexTrafficEnv  
from mutilDqfsAgent import ExpertDQN 

# --- 1. 路径自动兼容处理 ---
# 确保无论在本地还是云端，都能找到同级目录下的模型和数据
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mutilDqfd_final_rl_model.pth")

# --- 2. 页面基础设置 ---
st.set_page_config(page_title="EV RL Simulator", layout="wide", page_icon="🔋")
st.title("🔋 伦敦电动车多智能体补能调度可视化平台")
st.markdown("---")

# --- 3. 侧边栏：实验参数控制 ---
with st.sidebar:
    st.header("⚙️ 仿真参数配置")
    num_agents = st.slider("智能体数量 (Agents)", 1, 20, 10)
    soc_limit = st.slider("充电决策阈值 (SoC %)", 5.0, 50.0, 25.0)
    weather_val = st.select_slider("天气影响因子", options=[0.8, 1.0, 1.2, 1.5, 2.0], value=1.2)
    test_episodes = st.number_input("评估局数 (Episodes)", min_value=1, max_value=100, value=20)
    
    st.divider()
    run_button = st.button("🚀 开始运行评估测试", use_container_width=True)

# --- 4. 模型加载逻辑 (带缓存) ---
@st.cache_resource
def load_assets(_num_agents):
    # 初始化环境
    env = HexTrafficEnv(num_agents=_num_agents)
    # 初始化 Agent (根据你之前的 state_size=20)
    agent = ExpertDQN(state_size=20, action_size=6, device="cpu") 
    
    if os.path.exists(MODEL_PATH):
        try:
            agent.policy_net.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
            st.sidebar.success("✅ 已成功加载模型权重")
        except Exception as e:
            st.sidebar.error(f"❌ 模型加载失败: {e}")
    else:
        st.sidebar.warning("⚠️ 未找到模型文件，将使用随机权重")
        
    return env, agent

# --- 5. 主程序逻辑 ---
env, agent = load_assets(num_agents)
env.soc_threshold = soc_limit
env.weather_factor = weather_val

if run_button:
    # 进度条和状态显示
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 统计数据初始化
    results = {
        "success": 0,
        "timeout": 0,
        "no_battery": 0,
        "out_of_road": 0,
        "final_soc": []
    }

    # 执行评估循环
    for ep in range(test_episodes):
        status_text.text(f"正在进行第 {ep+1}/{test_episodes} 局模拟...")
        obs, _ = env.reset()
        done = False
        
        while not done:
            actions = []
            for i in range(num_agents):
                # 调用你的模型进行预测 (非训练模式)
                act = agent.select_action(obs[i], env, i, training=False)
                actions.append(act)
            
            obs, rewards, all_done, truncated, infos = env.step(actions)
            done = all_done or truncated
        
        # 统计本局结果
        for info in infos:
            reason = info.get("reason", "")
            if reason == "success":
                results["success"] += 1
            elif reason == "out_of_battery":
                results["no_battery"] += 1
            elif reason == "timeout_penalty":
                results["timeout"] += 1
            elif reason == "out_of_road":
                results["out_of_road"] += 1
            
            if "soc" in info:
                results["final_soc"].append(info["soc"])

        # 更新进度条
        progress_bar.progress((ep + 1) / test_episodes)

    status_text.success(f"✅ 测试完成！共计运行 {test_episodes} 局。")

    # --- 6. 结果展示面板 ---
    total_samples = test_episodes * num_agents
    success_rate = (results["success"] / total_samples) * 100
    avg_soc = np.mean(results["final_soc"]) if results["final_soc"] else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("成功率 (Success)", f"{success_rate:.1f}%")
    col2.metric("平均剩余电量", f"{avg_soc:.1f}%")
    col3.metric("断电次数", results["no_battery"])
    col4.metric("超时次数", results["timeout"])

    st.divider()

    # --- 7. 可视化图表绘制 ---
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("📊 任务结果分布")
        # 准备饼图数据
        dist_data = pd.DataFrame({
            "状态": ["Success", "No Battery", "Timeout", "Crash"],
            "数量": [results["success"], results["no_battery"], results["timeout"], results["out_of_road"]]
        })
        st.bar_chart(dist_data.set_index("状态"))

    with chart_col2:
        st.subheader("⚡ 最终电量分布 (SoC)")
        if results["final_soc"]:
            soc_df = pd.DataFrame(results["final_soc"], columns=["Final SoC"])
            st.area_chart(soc_df)
        else:
            st.write("暂无电量数据")

else:
    # 默认未运行时的欢迎页面
    st.info("💡 请在左侧调整参数后，点击‘开始运行评估测试’按钮来查看模型表现。")
    st.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", width=200)