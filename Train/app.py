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

# --- 1. 路径兼容性处理 ---
# 自动定位当前脚本所在目录，确保在云端也能找到模型和地图
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mutilDqfd_final_rl_model.pth")

# --- 2. 页面基础设置 ---
st.set_page_config(page_title="EV RL Simulator", layout="wide", page_icon="🔋")
st.title("🔋 伦敦电动车多智能体补能调度可视化平台")
st.markdown("""
本平台用于展示经过 **DQfD (Deep Q-Learning from Demonstrations)** 训练后的多智能体系统在复杂城市路网中的补能决策表现。
""")
st.markdown("---")

# --- 3. 侧边栏：实验参数控制 ---
with st.sidebar:
    st.header("⚙️ 仿真参数配置")
    num_agents = st.slider("智能体数量 (Agents)", 1, 20, 10)
    soc_limit = st.slider("充电决策阈值 (SoC %)", 5.0, 50.0, 20.0)
    weather_val = st.select_slider("天气影响因子", options=[0.8, 1.0, 1.2, 1.5, 2.0], value=1.2)
    test_episodes = st.number_input("评估局数 (Episodes)", min_value=1, max_value=100, value=20)
    
    st.divider()
    st.info("模型配置: Phase 7.1 (Tanh + Gated SoC)")
    run_button = st.button("🚀 开始运行评估测试", use_container_width=True)

# --- 4. 资源加载逻辑 (带缓存优化) ---
@st.cache_resource
def load_assets(_num_agents):
    # 初始化环境
    env = HexTrafficEnv(num_agents=_num_agents)
    # 初始化 Agent (根据你定义的 20 维状态空间)
    agent = ExpertDQN(state_size=20, action_size=6, device="cpu") 
    
    if os.path.exists(MODEL_PATH):
        try:
            agent.policy_net.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
            st.sidebar.success("✅ 模型权重加载成功")
        except Exception as e:
            st.sidebar.error(f"❌ 权重加载失败: {e}")
    else:
        st.sidebar.warning("⚠️ 未找到模型文件，将使用随机初始化")
        
    return env, agent

# 加载环境和模型
env, agent = load_assets(num_agents)
env.soc_threshold = soc_limit
env.weather_factor = weather_val

# --- 5. 主程序执行逻辑 ---
if run_button:
    # 进度监控组件
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 数据统计容器
    stats = {
        "success_count": 0,
        "timeout_count": 0,
        "no_battery_count": 0,
        "crash_count": 0,
        "final_soc_list": [],
        "reward_history": []
    }

    # 执行仿真循环
    for ep in range(test_episodes):
        status_text.text(f"正在模拟第 {ep+1}/{test_episodes} 局...")
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            actions = []
            for i in range(num_agents):
                # 调用模型预测动作 (关闭训练模式)
                act = agent.select_action(obs[i], env, i, training=False)
                actions.append(act)
            
            obs, rewards, all_done, truncated, infos = env.step(actions)
            ep_reward += sum(rewards) / num_agents
            done = all_done or truncated
        
        # 记录每局后的智能体状态
        for info in infos:
            reason = info.get("reason", "")
            if reason == "success":
                stats["success_count"] += 1
            elif reason == "out_of_battery":
                stats["no_battery_count"] += 1
            elif reason == "timeout_penalty":
                stats["timeout_count"] += 1
            elif reason == "out_of_road":
                stats["crash_count"] += 1
            
            if "soc" in info:
                stats["final_soc_list"].append(info["soc"])
        
        stats["reward_history"].append(ep_reward)
        progress_bar.progress((ep + 1) / test_episodes)

    status_text.success(f"🎉 评估完成！共测试 {test_episodes * num_agents} 个样本点。")

    # --- 6. 结果展示看板 ---
    total_samples = test_episodes * num_agents
    sr = (stats["success_count"] / total_samples) * 100
    avg_soc = np.mean(stats["final_soc_list"]) if stats["final_soc_list"] else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("成功率 (SR)", f"{sr:.1f}%", delta="Target: 60%")
    m2.metric("平均剩余电量", f"{avg_soc:.1f}%")
    m3.metric("断电失败次数", stats["no_battery_count"])
    m4.metric("超时次数", stats["timeout_count"])

    st.divider()

    # --- 7. 数据可视化分析 ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📌 任务状态分布 (Task Distribution)")
        dist_df = pd.DataFrame({
            "Category": ["Success", "No Battery", "Timeout", "Crash"],
            "Count": [stats["success_count"], stats["no_battery_count"], stats["timeout_count"], stats["crash_count"]]
        })
        st.bar_chart(dist_df.set_index("Category"))

    with col_right:
        st.subheader("📉 最终电量分布 (SoC Density)")
        if stats["final_soc_list"]:
            soc_df = pd.DataFrame(stats["final_soc_list"], columns=["SoC"])
            st.area_chart(soc_df)
        else:
            st.write("暂无有效电量数据")

    # 底部详细数据表
    with st.expander("查看原始统计数据"):
        st.write(pd.DataFrame([stats]))

else:
    # 初始欢迎画面
    st.info("👈 请在左侧侧边栏配置仿真参数，然后点击‘开始运行评估测试’。")
    st.caption("提示：增加智能体数量会模拟更高密度的交通压力。")