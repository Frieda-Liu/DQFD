import torch
import numpy as np
from EnvNew import HexTrafficEnv
from DQNAgent import ExpertDQN
from collections import Counter

def run_batch_test(num_episodes=200, weather=1.2):
    env = HexTrafficEnv(radius=120)
    agent = ExpertDQN(env.observation_space.shape[0], env.action_space.n)
    
    # 加载模型
    agent.policy_net.load_state_dict(torch.load("dqfd_final_rl_model.pth", map_location='cpu'))
    agent.policy_net.eval()
    
    success_count = 0
    total_reward = 0
    failure_reasons = []
    final_socs = []
    steps_list = []

    print(f"🚀 开始批量测试: 运行 {num_episodes} 次 | 天气系数: {weather}")

    for ep in range(num_episodes):
        state, _ = env.reset()
        env.weather_factor = weather # 统一天气进行压测
        done = False
        ep_reward = 0
        
        while not done:
            with torch.no_grad():
                action = agent.select_action(state, env, training=False)
            
            next_state, reward, term, trunc, info = env.step(action)
            state = next_state
            ep_reward += reward
            done = term or trunc
        
        # 统计数据
        if info.get("success"):
            success_count += 1
            final_socs.append(env.vehicle.soc)
            steps_list.append(env.step_count)
        else:
            failure_reasons.append(info.get("reason", "unknown"))
            
        total_reward += ep_reward
        if (ep + 1) % 50 == 0:
            print(f"已完成 {ep + 1}/{num_episodes}...")

    # --- 输出结果报告 ---
    print("\n" + "="*30)
    print(f"📊 批量测试报告 (天气: {weather})")
    print(f"✅ 总成功率: {success_count / num_episodes:.2%}")
    if success_count > 0:
        print(f"🔋 成功局平均剩余电量: {np.mean(final_socs):.2f}%")
        print(f"🏃 成功局平均步数: {np.mean(steps_list):.1f}")
    print(f"❌ 失败原因统计: {Counter(failure_reasons)}")
    print("="*30)

if __name__ == "__main__":
    # 你可以分别跑 1.0, 1.5, 2.0 看看成功率的变化曲线
    run_batch_test(num_episodes=200, weather=1.2)