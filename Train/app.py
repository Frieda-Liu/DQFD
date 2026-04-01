import streamlit as st
import pandas as pd
import torch
import numpy as np
import os
import time

# 导入你自己的模块
from mutilEnv import HexTrafficEnv  
from mutilDqfsAgent import ExpertDQN 

# --- 1. 路径兼容性处理 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 确认你的模型文件名，根据截图是 refined_final_model.pth
MODEL_FILENAME = "refined_final_model.pth" 
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)

# --- 2. 页面基础设置 ---
st.set_page_config(page_title="EV RL Simulator", layout="wide", page_icon="🔋")
st.title("🔋 伦敦电动车多智能体补能调度可视化平台")
st.markdown(f"当前运行模式：**模型推理 + 动态专家干预** | 加载权重：`{MODEL_FILENAME}`")

# --- 3. 侧边栏：参数调整 ---
with st.sidebar:
    st.header("⚙️ 仿真参数配置")
    num_agents = st.slider("智能体数量 (Agents)", 1, 20, 10)
    soc_limit = st.slider("充电决策阈值 (SoC %)", 5.0, 50.0, 25.0)
    
    st.divider()
    st.header("🧠 决策权重调节")
    # --- 新增：专家权重滑块 ---
    expert_w = st.slider("专家干预权重 (Expert Weight)", 0.0, 1.0, 0.2, help="0.0 为纯 AI 决策，1.0 为纯专家启发式算法")
    weather_val = st.select_slider("天气因子 (Weather)", options=[0.8, 1.0, 1.2, 1.5, 2.0], value=1.2)
    test_episodes = st.number_input("评估局数 (Episodes)", min_value=1, max_value=50, value=10)
    
    st.divider()
    run_button = st.button("🚀 开始运行评估测试", use_container_width=True)

# --- 4. 资源加载 (带缓存) ---
@st.cache_resource
def load_assets(_num_agents):
    env = HexTrafficEnv(num_agents=_num_agents)
    # 注意参数名 state_dim, action_dim 必须匹配 DQNAgent.py
    agent = ExpertDQN(state_dim=20, action_dim=6) 
    
    if os.path.exists(MODEL_PATH):
        try:
            agent.policy_net.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
            st.sidebar.success("✅ 模型加载成功")
        except Exception as e:
            st.sidebar.error(f"❌ 加载失败: {e}")
    else:
        st.sidebar.warning(f"⚠️ 找不到文件: {MODEL_FILENAME}")
        
    return env, agent

env, agent = load_assets(num_agents)
# 同步侧边栏参数到环境
env.soc_threshold = soc_limit
env.weather_factor = weather_val

# --- 5. 执行模拟逻辑 ---
if run_button:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    stats = {
        "success": 0, "timeout": 0, "no_battery": 0, "crash": 0,
        "final_soc": [], "rewards": []
    }

    for ep in range(int(test_episodes)):
        status_text.text(f"正在模拟第 {ep+1}/{test_episodes} 局...")
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            actions = []
            for i in range(num_agents):
                # --- 关键修改：传入 expert_weight 参数 ---
                # 注意：这里 training 设为 True 是为了激活 select_action 内部的专家随机逻辑
                # 但我们把 epsilon 设为 0 以关闭随机探索
                agent.epsilon = 0.0 
                act = agent.select_action(obs[i], env, agent_id=i, training=True, expert_weight=expert_w)
                actions.append(act)
            
            obs, rewards, all_done, truncated, infos = env.step(actions)
            ep_reward += sum(rewards) / num_agents
            done = all_done or truncated
        
        # 记录结果
        for info in infos:
            res = info.get("reason", "")
            if res == "success": stats["success"] += 1
            elif res == "out_of_battery": stats["no_battery"] += 1
            elif res == "timeout_penalty": stats["timeout"] += 1
            elif res == "out_of_road": stats["crash"] += 1
            if "soc" in info: stats["final_soc"].append(info["soc"])
            
        stats["rewards"].append(ep_reward)
        progress_bar.progress((ep + 1) / test_episodes)

    status_text.success("✅ 模拟运行结束！")

    # --- 6. 结果面板 ---
    total = test_episodes * num_agents
    sr = (stats["success"] / total) * 100
    avg_soc = np.mean(stats["final_soc"]) if stats["final_soc"] else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("成功率 (SR)", f"{sr:.1f}%")
    col2.metric("平均剩余 SoC", f"{avg_soc:.1f}%")
    col3.metric("电量耗尽次数", stats["no_battery"])
    col4.metric("超时次数", stats["timeout"])

    st.divider()

    # --- 7. 数据分析图表 ---
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📌 任务结果分布")
        res_df = pd.DataFrame({
            "Status": ["Success", "No Battery", "Timeout", "Crash"],
            "Count": [stats["success"], stats["no_battery"], stats["timeout"], stats["crash"]]
        })
        st.bar_chart(res_df.set_index("Status"))

    with c2:
        st.subheader("📈 终端 SoC 累计分布")
        if stats["final_soc"]:
            st.area_chart(pd.DataFrame(stats["final_soc"], columns=["Final SoC"]))
        else:
            st.write("No SoC Data")

    with st.expander("点击查看原始数据"):
        st.write(stats)

else:
    st.info("👈 调整左侧参数（包括专家权重），然后点击运行按钮。")