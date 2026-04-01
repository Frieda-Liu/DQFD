import streamlit as st
import pandas as pd
import torch
import numpy as np
import os
import time
import matplotlib.pyplot as plt

# 导入你的自定义模块
from mutilEnv import HexTrafficEnv  
from mutilDqfsAgent import ExpertDQN 

# --- 1. 路径兼容性处理 (Path Setup) ---
# 获取当前 Train 文件夹的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 根据你的截图，模型文件在 Train 文件夹内
# 请确保文件名与 GitHub 上的完全一致 (例如 "refined_final_model.pth")
MODEL_FILENAME = "refined_final_model.pth" 
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)

# --- 2. 页面基础设置 (Page Config) ---
st.set_page_config(page_title="EV RL Simulator", layout="wide", page_icon="🔋")
st.title("🔋 伦敦电动车多智能体补能调度可视化平台")
st.markdown(f"""
本平台展示基于 **DQfD** 算法的智能调度结果。当前加载模型：`{MODEL_FILENAME}`
""")
st.markdown("---")

# --- 3. 侧边栏：参数控制 (Sidebar) ---
with st.sidebar:
    st.header("⚙️ 仿真参数配置")
    num_agents = st.slider("智能体数量 (Agents)", 1, 20, 10)
    soc_limit = st.slider("充电决策阈值 (SoC %)", 5.0, 50.0, 20.0)
    weather_val = st.select_slider("天气影响因子", options=[0.8, 1.0, 1.2, 1.5, 2.0], value=1.2)
    test_episodes = st.number_input("评估局数 (Episodes)", min_value=1, max_value=100, value=10)
    
    st.divider()
    st.info("Strategy: Tanh Scaling + Gated SoC Control")
    run_button = st.button("🚀 开始运行评估测试", use_container_width=True)

# --- 4. 资源加载 (Resource Loading) ---
@st.cache_resource
def load_assets(_num_agents):
    # 初始化环境 (mutilEnv 内部会自动处理 MAP/ 路径)
    env = HexTrafficEnv(num_agents=_num_agents)
    
    # 初始化 Agent (使用正确的参数名: state_dim, action_dim)
    agent = ExpertDQN(state_dim=20, action_dim=6) 
    
    if os.path.exists(MODEL_PATH):
        try:
            # 加载权重
            state_dict = torch.load(MODEL_PATH, map_location='cpu')
            agent.policy_net.load_state_dict(state_dict)
            st.sidebar.success("✅ 模型权重加载成功")
        except Exception as e:
            st.sidebar.error(f"❌ 权重加载失败: {e}")
    else:
        st.sidebar.warning(f"⚠️ 未找到文件: {MODEL_FILENAME}")
        
    return env, agent

# 执行加载
env, agent = load_assets(num_agents)
# 动态同步用户在侧边栏修改的参数
env.soc_threshold = soc_limit
env.weather_factor = weather_val

# --- 5. 仿真执行逻辑 (Simulation Loop) ---
if run_button:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 数据统计字典
    stats = {
        "success": 0, "timeout": 0, "no_battery": 0, "crash": 0,
        "soc_end": [], "rewards": []
    }

    for ep in range(int(test_episodes)):
        status_text.text(f"正在模拟第 {ep+1}/{test_episodes} 局...")
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            actions = []
            for i in range(num_agents):
                # 预测动作 (training=False)
                act = agent.select_action(obs[i], env, i, training=False)
                actions.append(act)
            
            obs, rewards, all_done, truncated, infos = env.step(actions)
            ep_reward += sum(rewards) / num_agents
            done = all_done or truncated
        
        # 统计本局每个 Agent 的最终状态
        for info in infos:
            res = info.get("reason", "")
            if res == "success": stats["success"] += 1
            elif res == "out_of_battery": stats["no_battery"] += 1
            elif res == "timeout_penalty": stats["timeout"] += 1
            elif res == "out_of_road": stats["crash"] += 1
            
            if "soc" in info: stats["soc_end"].append(info["soc"])
        
        stats["rewards"].append(ep_reward)
        progress_bar.progress((ep + 1) / test_episodes)

    status_text.success(f"✅ 评估完成！共分析了 {test_episodes * num_agents} 个样本。")

    # --- 6. 结果看板 (Dashboard Metrics) ---
    total = test_episodes * num_agents
    sr = (stats["success"] / total) * 100
    avg_soc = np.mean(stats["soc_end"]) if stats["soc_end"] else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("成功率 (SR)", f"{sr:.1f}%")
    col2.metric("平均剩余 SoC", f"{avg_soc:.1f}%")
    col3.metric("电量耗尽次数", stats["no_battery"])
    col4.metric("超时次数", stats["timeout"])

    st.divider()

    # --- 7. 图表展示 (Visualizations) ---
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("📌 失败原因分析 (Failure Analysis)")
        chart_df = pd.DataFrame({
            "Status": ["Success", "No Battery", "Timeout", "Crash"],
            "Count": [stats["success"], stats["no_battery"], stats["timeout"], stats["crash"]]
        })
        st.bar_chart(chart_df.set_index("Status"))

    with c2:
        st.subheader("📉 终端电量分布 (Final SoC Distribution)")
        if stats["soc_end"]:
            st.area_chart(pd.DataFrame(stats["soc_end"], columns=["SoC"]))
        else:
            st.write("暂无数据")

    with st.expander("查看详细统计 JSON"):
        st.json(stats)

else:
    st.info("👈 请在左侧侧边栏设置参数，并点击‘开始运行’。")
    st.image("https://img.icons8.com/clouds/200/000000/lightning-bolt.png")