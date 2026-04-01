# DQNAgent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
from collections import deque
from mutilEnv import HexTrafficEnv
from expert import Expert  
import time
import pickle
import os

class ExpertDQN:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 网络
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        ).to(self.device)
        
        self.target_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0005)
        self.memory = deque(maxlen=50000)
        
        # 训练参数
        self.batch_size = 128
        self.gamma = 0.99
        self.epsilon = 0.5
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.target_update = 100

        self.expert = Expert()
        
        # 专家演示缓存
        self.expert_replay = deque(maxlen=200000)
        
        self.step_count = 0
    
    def collect_expert_demos(self, env, num_episodes=2000, n_step=10):
        print(f"\nstart collect expert data...")
        success_count = 0
        
        for ep in range(num_episodes):
            obs, _ = env.reset() 
            # dynamic weather
            env.weather_factor = 1.0 if ep < num_episodes * 0.3 else 1.3 if ep < num_episodes * 0.7 else 1.6
            
            for u, v, d in env.graph.edges(data=True):
                d['current_time'] = d.get('base_time', 10.0)
                d['speed'] = env.speed_map.get(v, 40.0) 

            # randomly make congestion
            all_edges = list(env.graph.edges())
            num_congested = int(len(all_edges) * random.uniform(0.05, 0.10))
            congested_edges = random.sample(all_edges, num_congested)
            
            for u, v in congested_edges:
                original_time = env.graph[u][v]['current_time']
                env.graph[u][v]['current_time'] = original_time * random.uniform(3.0, 8.0)
                env.graph[u][v]['speed'] = random.uniform(5.0, 15.0)

            temp_buffers = [[] for _ in range(env.num_agents)]
            agent_dones = [False] * env.num_agents
            
            while not all(agent_dones):
                actions = []
                for i in range(env.num_agents):
                    actions.append(self.expert.get_action(env, i) if not agent_dones[i] else 0)
                
                next_obs, rewards, all_done_flag, truncated_flag, infos = env.step(actions)

                for i in range(env.num_agents):
                    if agent_dones[i]: continue
                    v = env.vehicles[i]
                    # record current step
                    temp_buffers[i].append({
                        's': obs[i].copy(), 
                        'a': actions[i], 
                        'r': rewards[i], 
                        's_next': next_obs[i].copy(), 
                        'd': float(env.dones[i])
                    })
                    # when n-step or done 
                    if len(temp_buffers[i]) >= n_step or env.dones[i]:
                        is_success = v.goal
                        is_running = v.is_active
                        # calculate n-step reward
                        R_n = sum([(self.gamma ** idx) * step['r'] for idx, step in enumerate(temp_buffers[i])])
                        
                        first_step = temp_buffers[i][0]
                        last_step = temp_buffers[i][-1]
                        
                        # only record success case
                        if is_success or is_running:
                            self.expert_replay.append((
                                first_step['s'], first_step['a'], first_step['r'], 
                                first_step['s_next'], first_step['d'],
                                R_n, last_step['s_next'], last_step['d']
                            ))
                        
                        if not env.dones[i]:
                            temp_buffers[i].pop(0)
                        else:
                            temp_buffers[i].clear()
                            agent_dones[i] = True
                            # print(infos)
                            if is_success: 
                                success_count += 1

                obs = next_obs
                if env.step_count >= env.maxsteps: break
                
            if (ep + 1) % 50 == 0:
                print(f"process: {ep+1}/{num_episodes} | already saved: {len(self.expert_replay)} | successful: {success_count}")
            
    def pretrain_with_expert(self, epochs=200):
        dataset_size = len(self.expert_replay)
        # 将数据转为列表并打乱，保证覆盖率
        expert_list = list(self.expert_replay)
        
        for epoch in range(epochs):
            random.shuffle(expert_list) # 每个 epoch 重新打乱
            epoch_loss = 0
            correct_actions = 0
            
            for start in range(0, dataset_size, self.batch_size):
                end = min(start + self.batch_size, dataset_size)
                batch = expert_list[start:end]
                states, actions, rewards, next_states, dones, r_n_val, s_n_next_val, d_n_val = zip(*batch)
                states = torch.FloatTensor(np.array(states)).to(self.device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
                rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
                next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
                dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
                
                r_n = torch.FloatTensor(r_n_val).unsqueeze(1).to(self.device)
                s_n_next = torch.FloatTensor(np.array(s_n_next_val)).to(self.device)
                d_n = torch.FloatTensor(d_n_val).unsqueeze(1).to(self.device)

                q_values = self.policy_net(states)
                current_q = q_values.gather(1, actions) 
                
                # TD Targets (1-step & n-step)
                with torch.no_grad():
                    next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
                    target_q = rewards + (1 - dones) * self.gamma * next_q
                    
                    next_q_n = self.target_net(s_n_next).max(1)[0].unsqueeze(1)
                    target_q_n = r_n + (1 - d_n) * (self.gamma ** 10) * next_q_n

                # 1. 1-step MSE
                loss_dq = nn.MSELoss()(current_q, target_q)
                # 2. n-step MSE
                loss_n = nn.MSELoss()(current_q, target_q_n)
                # 3. Large Margin Supervised Loss (JE)
                l = torch.ones_like(q_values) * 0.8
                l.scatter_(1, actions, 0)
                max_q_with_margin = (q_values + l).max(1)[0]
                loss_je = (max_q_with_margin - current_q.squeeze()).mean()
                
                # 4. L2 Reg
                l2_reg = sum(p.pow(2.0).sum() for p in self.policy_net.parameters())

                # Total Combined Loss
                batch_loss = loss_dq + 1.0*loss_n + 1.0*loss_je + 1e-5*l2_reg
                
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
                
                epoch_loss += batch_loss.item()
                correct_actions += (q_values.argmax(1).unsqueeze(1) == actions).sum().item()

            accuracy = correct_actions / dataset_size
            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / (dataset_size / self.batch_size)
                print(f"Epoch [{epoch+1}/{epochs}] | Avg Loss: {avg_loss:.4f} | Acc: {accuracy:.2%}")
                self.target_net.load_state_dict(self.policy_net.state_dict())

    def select_action(self, state, env, agent_id=0, training=True, expert_weight=0.5):
        # 1) 专家诱导：传 agent_id 给专家
        if training and random.random() < expert_weight:
            return self.expert.get_action(env, agent_id=agent_id)

        # 2) 随机探索 
        if training and random.random() < self.epsilon:
            mask = state[8:14]
            legal_actions = [i for i, m in enumerate(mask) if m >= 0.5]
            if legal_actions:
                return random.choice(legal_actions)
            return random.randint(0, 5)
        
        # 3) 网络预测
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.policy_net(state_tensor)[0].cpu().numpy()

        mask = state[8:14] 
        q[mask < 0.5] = -1e9
        return int(np.argmax(q))
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def update_hybrid(self):
        if len(self.memory) < self.batch_size:
            return 0.0

        # 1. sampling
        expert_batch = random.sample(self.expert_replay, self.batch_size // 2)
        s_e, a_e, r_e, sn_e, d_e, rn_e, snn_e, dn_e = zip(*expert_batch)

        # 2. Expert: transfer to Tensor 
        s_e = torch.FloatTensor(np.array(s_e)).to(self.device)
        a_e = torch.LongTensor(a_e).unsqueeze(1).to(self.device)
        r_e = torch.FloatTensor(r_e).unsqueeze(1).to(self.device)
        sn_e = torch.FloatTensor(np.array(sn_e)).to(self.device)
        d_e = torch.FloatTensor(d_e).unsqueeze(1).to(self.device)
        rn_e = torch.FloatTensor(rn_e).unsqueeze(1).to(self.device)
        snn_e = torch.FloatTensor(np.array(snn_e)).to(self.device)
        dn_e = torch.FloatTensor(dn_e).unsqueeze(1).to(self.device)

        q_e = self.policy_net(s_e)
        current_q_e = q_e.gather(1, a_e)
        
        with torch.no_grad():
            next_q_e = self.target_net(sn_e).max(1)[0].unsqueeze(1)
            target_q_e = r_e + (1 - d_e) * self.gamma * next_q_e 
            
            next_q_n_e = self.target_net(snn_e).max(1)[0].unsqueeze(1)
            target_q_n_e = rn_e + (1 - dn_e) * (self.gamma ** 10) * next_q_n_e

        loss_dq_e = nn.MSELoss()(current_q_e, target_q_e)
        loss_n_e = nn.MSELoss()(current_q_e, target_q_n_e) 
        
        # JE
        margin = 0.8
        l = torch.ones_like(q_e) * margin
        l.scatter_(1, a_e, 0)
        max_q_with_margin = (q_e + l).max(1)[0]
        loss_je = (max_q_with_margin - current_q_e.squeeze()).mean()

        # JL2
        l2_reg = torch.tensor(0.).to(self.device)
        for param in self.policy_net.parameters():
            l2_reg += torch.norm(param)

        # 3. Own loss
        self_batch = random.sample(self.memory, self.batch_size // 2)
        s_s, a_s, r_s, sn_s, d_s = zip(*self_batch)
        
        s_s = torch.FloatTensor(np.array(s_s)).to(self.device)
        a_s = torch.LongTensor(a_s).unsqueeze(1).to(self.device)
        r_s = torch.FloatTensor(r_s).unsqueeze(1).to(self.device)
        sn_s = torch.FloatTensor(np.array(sn_s)).to(self.device)
        d_s = torch.FloatTensor(d_s).unsqueeze(1).to(self.device)
        
        curr_q_s = self.policy_net(s_s).gather(1, a_s)
        with torch.no_grad():
            next_q_s = self.target_net(sn_s).max(1)[0].unsqueeze(1)
            target_q_s = r_s + (1 - d_s) * self.gamma * next_q_s
        loss_dq_self = nn.MSELoss()(curr_q_s, target_q_s)
        
        # all
        # lambda_je = max(0.5, 3.0 * (0.9999 ** self.step_count))
        lambda_je = max(1.0, 5.0 * (0.99995 ** self.step_count))
        total_loss = loss_dq_e + loss_n_e + lambda_je * loss_je + loss_dq_self + 1e-5 * l2_reg

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return total_loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def rl_train(env, agent):
    try:
       
        agent.policy_net.load_state_dict(torch.load("mutildqfd_pretrained_10agent_model.pth"))
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        print("load successful")
    except:
        print("load fail")

    total_episodes = 3000
    agent.epsilon = 0.1 
    
    history_rewards = []
    history_losses = [] 
    
    
    recent_avg = 0.0 

    for ep in range(total_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        ep_losses = []
        
        agent.decay_epsilon() 
        
        target_score = 550.0 
        if len(history_rewards) > 0:
            recent_avg = sum(history_rewards[-10:]) / len(history_rewards[-10:])

        # --- dynamic Expert Weight ---
        if ep < 100:
            cur_expert_w = max(0.4, 0.8 * (0.995 ** ep))
        else:
            if recent_avg > target_score:
                cur_expert_w = max(0.05, cur_expert_w - 0.05)
            elif recent_avg < 300: 
                cur_expert_w = min(0.5, cur_expert_w + 0.1)
            else:
                cur_expert_w = max(0.1, cur_expert_w * 0.99)

        while not done:
            actions = []
            for i in range(env.num_agents):
                act = agent.select_action(obs[i], env, agent_id=i, training=True, expert_weight=cur_expert_w)
                actions.append(act)
            
            next_obs, rewards, all_done, truncated, infos = env.step(actions)
            
            step_avg_reward = sum(rewards) / env.num_agents
            ep_reward += step_avg_reward  
            
            for i in range(env.num_agents):
                # filter out bad action 
                if rewards[i] > -50: 
                    agent_done = env.dones[i]
                    agent.store_transition(obs[i], actions[i], rewards[i], next_obs[i], float(agent_done))
            
            # -get loss
            loss = agent.update_hybrid() 
            if loss is not None:
                ep_losses.append(loss)
            
            obs = next_obs
            done = all_done or truncated
        
        # record loss and acc
        history_rewards.append(ep_reward)
        if len(ep_losses) > 0:
            history_losses.append(sum(ep_losses) / len(ep_losses))
        else:
            history_losses.append(0.0)

        if (ep + 1) % 10 == 0:
            avg_reward = sum(history_rewards[-10:]) / 10
            avg_loss = sum(history_losses[-10:]) / 10 # <--- 新增：计算平均 Loss
            print(f"Episode {ep+1}/{total_episodes} | Avg Reward: {avg_reward:.2f} | Avg Loss: {avg_loss:.4f} | Expert_W: {cur_expert_w:.2f} | Eps: {agent.epsilon:.3f}")

    torch.save(agent.policy_net.state_dict(), "mutilDqfd_final_rl_model.pth")
    print("model saved as mutilDqfd_final_rl_model.pth")
    
    # 建议返回这两个列表，方便之后画图
    return history_rewards, history_losses
def train_phase_2(env, agent, total_episodes=300, checkpoint_path="mutilDqfd_final_rl_model.pth"):
    # 1. load previces model
    if os.path.exists(checkpoint_path):
    # 💡 这里的 map_location 是关键，它会把原本属于 GPU 的参数强制拉到 CPU 上
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        agent.policy_net.load_state_dict(state_dict)
        agent.target_net.load_state_dict(state_dict)
        print(f"✅ 已成功在 CPU 上加载权重: {checkpoint_path}")
    else:
        print("cannot find ",checkpoint_path )

  
    initial_expert_w = 0.4  
    final_expert_w = 0.05   
    history_rewards = []

    agent.policy_net.train()

    for ep in range(total_episodes):
        cur_expert_w = max(final_expert_w, initial_expert_w - (initial_expert_w - final_expert_w) * (ep / total_episodes))
        
        obs, _ = env.reset()
        ep_reward = 0
        done = False
        
        while not done:
            actions = []
            for i in range(env.num_agents):
                act = agent.select_action(obs[i], env, agent_id=i, training=True, expert_weight=cur_expert_w)
                actions.append(act)

            # 执行环境步进
            next_obs, rewards, all_done, truncated, infos = env.step(actions)
            
            # --- 核心修复：逐个 Agent 存储并添加额外惩罚 ---
            for i in range(env.num_agents):
                r = rewards[i]
                
                if env.agent_positions[i] in env.visited_nodes[i]:
                    r -= 10.0 
                if env.vehicles[i].goal:
                    r += 50.0

                agent.store_transition(obs[i], actions[i], r, next_obs[i], float(env.dones[i]))
           
            agent.update_hybrid()

            obs = next_obs
            ep_reward += sum(rewards) / env.num_agents # 记录原始奖励，方便观察真实趋势
            done = all_done or truncated

        history_rewards.append(ep_reward)
        
        
        if (ep + 1) % 10 == 0:
            avg_r = np.mean(history_rewards[-10:])
            print(f"Episode {ep+1}/{total_episodes} | Avg Reward: {avg_r:.2f} | Expert_W: {cur_expert_w:.3f}")

    # 3. 保存最终脱产模型
    torch.save(agent.policy_net.state_dict(), "mutilDqfd_Phase2_Final.pth")
    print("Phase 2 complete. saved as mutilDqfd_Phase2_Final.pth")
def train_phase_3_recovery(env, agent, total_episodes=200):
    agent.policy_net.load_state_dict(torch.load("mutilDqfd_Phase2_Final.pth", map_location='cpu'))
    
    initial_expert_w = 0.2  # 固定在 0.2，给点“速效救心丸”
    history_rewards = []

    for ep in range(total_episodes):
        obs, _ = env.reset()
        ep_reward = 0
        done = False
        
        while not done:
            actions = []
            for i in range(env.num_agents):
                act = agent.select_action(obs[i], env, agent_id=i, training=True, expert_weight=initial_expert_w)
                actions.append(act)

            next_obs, rewards, all_done, truncated, infos = env.step(actions)
            
            for i in range(env.num_agents):
                r = rewards[i]
                # 1. 降低重复惩罚
                if env.agent_positions[i] in env.visited_nodes[i]:
                    r -= 3.0 
                # 2. 巨额终点奖励
                if env.vehicles[i].goal:
                    r += 200.0
                
                agent.store_transition(obs[i], actions[i], r, next_obs[i], float(env.dones[i]))
            
            agent.update_hybrid()
            obs = next_obs
            ep_reward += sum(rewards) / env.num_agents
            done = all_done or truncated

        if (ep + 1) % 10 == 0:
            print(f"Recovery Ep {ep+1} | Avg Reward: {np.mean(history_rewards[-10:]):.2f}")

    torch.save(agent.policy_net.state_dict(), "mutilDqfd_Final_Confidence.pth")
def train_phase4(env, agent, total_episodes=200):
    agent.policy_net.load_state_dict(torch.load("mutilDqfd_ULTRON_Final.pth", map_location='cuda'))
    history_rewards = []

    for ep in range(total_episodes):
        obs, _ = env.reset()
        current_w = 0.05 
        done = False
        ep_reward = 0
        

        last_distances = [env._get_hex_dist(env.agent_positions[i], env.target_positions[i]) for i in range(env.num_agents)]
        
        while not done:
            actions = []
            for i in range(env.num_agents):
                act = agent.select_action(obs[i], env, agent_id=i, training=True, expert_weight=current_w)
                actions.append(act)

            next_obs, rewards, all_done, truncated, infos = env.step(actions)
            
            for i in range(env.num_agents):
                pos = env.agent_positions[i]
                target = env.target_positions[i]
                curr_dist = env._get_hex_dist(pos, target)
                if curr_dist < last_distances[i]:
                    rewards[i] += 10.0 
                elif curr_dist > last_distances[i]:
                    rewards[i] -= 10.0  
                last_distances[i] = curr_dist
                if env.vehicles[i].goal:
                    rewards[i] += 800.0
                

                rewards[i] -= 0.5

                agent.store_transition(obs[i], actions[i], rewards[i], next_obs[i], float(env.dones[i]))
            
            agent.update_hybrid()
            obs = next_obs
            ep_reward += sum(rewards) / env.num_agents
            done = all_done or truncated

  
        history_rewards.append(ep_reward) 

        if (ep + 1) % 10 == 0:
            avg_r = np.mean(history_rewards[-10:])
            print(f"Refine Ep {ep+1} | Avg Reward: {avg_r:.2f}")

    torch.save(agent.policy_net.state_dict(), "refined_final_model.pth")
    print("✅ 训练完成，模型已保存为 refined_final_model.pth")
def train_phase5(env, agent, total_episodes=200):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent.policy_net.load_state_dict(torch.load("refined_final_model.pth", map_location=device))
    history_rewards = []

    for ep in range(total_episodes):
        obs, _ = env.reset()
        current_w = 0.02
        done = False
        ep_reward = 0
        

        last_distances = [env._get_hex_dist(env.agent_positions[i], env.target_positions[i]) for i in range(env.num_agents)]
        
        while not done:
            actions = []
            for i in range(env.num_agents):
                act = agent.select_action(obs[i], env, agent_id=i, training=True, expert_weight=current_w)
                actions.append(act)

            next_obs, rewards, all_done, truncated, infos = env.step(actions)
            
            for i in range(env.num_agents):
                pos = env.agent_positions[i]
                target = env.target_positions[i]
                curr_dist = env._get_hex_dist(pos, target)
                if curr_dist < last_distances[i]:
                    rewards[i] += 10.0 
                elif curr_dist > last_distances[i]:
                    rewards[i] -= 10.0  
                last_distances[i] = curr_dist
                if env.vehicles[i].goal:
                    rewards[i] += 800.0
                

                rewards[i] -= 1

                agent.store_transition(obs[i], actions[i], rewards[i], next_obs[i], float(env.dones[i]))
            
            agent.update_hybrid()
            obs = next_obs
            ep_reward += sum(rewards) / env.num_agents
            done = all_done or truncated

  
        history_rewards.append(ep_reward) 

        if (ep + 1) % 10 == 0:
            avg_r = np.mean(history_rewards[-10:])
            print(f"Refine Ep {ep+1} | Avg Reward: {avg_r:.2f}")

    torch.save(agent.policy_net.state_dict(), "phase5.pth")
    print("finished, saved as refined_final_model.pth")
def train_phase6(env, agent, total_episodes=200):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent.policy_net.load_state_dict(torch.load("refined_final_model.pth", map_location=device))
    
    history_rewards = []

    for ep in range(total_episodes):
        obs, _ = env.reset()
        # 专家权重动态衰减
        current_w = max(0.0, 0.02 * (1 - ep / (total_episodes * 0.8))) 
        done = False
        ep_reward = 0
        
        last_distances = [env._get_hex_dist(env.agent_positions[i], env.target_positions[i]) for i in range(env.num_agents)]
        
        while not done:
            actions = []
            for i in range(env.num_agents):
                act = agent.select_action(obs[i], env, agent_id=i, training=True, expert_weight=current_w)
                actions.append(act)

            next_obs, rewards, all_done, truncated, infos = env.step(actions)
            
            for i in range(env.num_agents):
                curr_dist = env._get_hex_dist(env.agent_positions[i], env.target_positions[i])
                dist_diff = last_distances[i] - curr_dist
                rewards[i] += dist_diff * 15.0 # 稍微加大移动奖励的敏感度
                last_distances[i] = curr_dist
                
                # --- B. 到达与生存奖励 (核心修改) ---
                if env.vehicles[i].goal:
                    rewards[i] += 1000.0 # 提高到达奖励
                
                if env.vehicles[i].is_dead: # 重点：如果没电或出路
                    rewards[i] -= 1200.0 # 严厉惩罚“死掉”，迫使它找桩
                
                # --- C. 充电正反馈 ---
                # 如果当前电量比上一时刻多，说明在充电
                if next_obs[i][4] > obs[i][4]: # 假设 obs 第4位是 SoC
                    rewards[i] += 30.0 

                agent.store_transition(obs[i], actions[i], rewards[i], next_obs[i], float(env.dones[i]))
            
            agent.update_hybrid()
            obs = next_obs
            ep_reward += sum(rewards) / env.num_agents
            done = all_done or truncated

        history_rewards.append(ep_reward) 
        if (ep + 1) % 10 == 0:
            avg_r = np.mean(history_rewards[-10:])
            print(f"Refine Ep {ep+1} | w:{current_w:.3f} | Avg R: {avg_r:.2f}")

    # 保存最终成品
    torch.save(agent.policy_net.state_dict(), "phase6.pth")
def train_phase7(env, agent):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    soup_path = "temp_soup_0.9559.pth"
    
    if os.path.exists(soup_path):
        agent.policy_net.load_state_dict(torch.load(soup_path, map_location=device))
        print(f"🚀 Loaded for Phase 7: {soup_path}")
    else:
        print(f"❌ Not found: {soup_path}")
        return

    learning_rate = 5e-6 
    for param_group in agent.optimizer.param_groups:
        param_group['lr'] = learning_rate

    total_episodes = 500 
    history_rewards = []
    success_count = 0

    for ep in range(total_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            actions = []
            for i in range(env.num_agents):
                action = agent.select_action(obs[i], env, agent_id=i, training=True, expert_weight=0)
                actions.append(action)
            
            next_obs, rewards, all_done, truncated, infos = env.step(actions)
          
            for i in range(env.num_agents):
                agent.store_transition(obs[i], actions[i], rewards[i], next_obs[i],float(env.dones[i]) )
            
    
            agent.update_hybrid()
            obs = next_obs
            ep_reward += sum(rewards) 
            
            done = all_done or truncated
            
            
            for info in infos:
                if info.get("success"):
                    success_count += 1

        history_rewards.append(ep_reward)
        
        if (ep + 1) % 10 == 0:
            avg_r = np.mean(history_rewards[-10:])
            sr = (success_count / ((ep + 1) * env.num_agents)) * 100
            print(f"Ep {ep+1}/{total_episodes} | AvgR: {avg_r:.2f} | Current SR: {sr:.1f}%")

    final_name = "phase7.pth"
    torch.save(agent.policy_net.state_dict(), final_name)
    print(f"✅ Finished! Final model saved as: {final_name}")

def run_collection_and_save(agent, env, filename="mutilexpert_data_nstep.pkl", num_episodes=100):
    """
    COLLECT EXPERT DATA
    """
    # # 1. check if already collected
    if os.path.exists(filename):
        print(f"warning: {filename} already exists")
        cont = input("recollect?(y/n): ")
        if cont.lower() != 'y':
            return
    agent.collect_expert_demos(env, num_episodes=num_episodes)

    with open(filename, 'wb') as f:
        pickle.dump(agent.expert_replay, f)
    
    print(f"expert data saved at {filename}")
    print(f"data amount: {len(agent.expert_replay)}")

def load_expert_data_to_agent(agent, filename="mutilexpert_data_nstep.pkl"):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
           
            agent.expert_replay = pickle.load(f)
        print(f"successful load, length: {len(agent.expert_replay)}")
        return True
    else:
        print(f"could not find {filename}")
        return False

if __name__ == "__main__":
    # 1. 初始化
    env = HexTrafficEnv(radius=120,num_agents=10)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = ExpertDQN(state_dim, action_dim) 

    ###################### collect expert data ##########################################
    # run_collection_and_save(agent, env, filename="mutilexpert_10agent_nstep.pkl", num_episodes=1000)

    # # check collected data
    # print("data check:")
    # actions = [d[1] for d in agent.expert_replay]
    # from collections import Counter
    # print(Counter(actions))

    ########################################## pre training ##########################################
    # "mutilexpert_data_nstep.pkl"
    # success = load_expert_data_to_agent(agent, "mutilexpert_10agent_nstep.pkl")
    
    # if success:
    #     # 2. start training
    #     print("pretraining...")
    #     agent.pretrain_with_expert(epochs=250)
        
    #     # 3. save model
    #     torch.save(agent.policy_net.state_dict(), "mutildqfd_pretrained_10agent_model.pth")
    #     print("pretrain saved")

    ########################### train rl #########################################################
    
    expert_data_path = "mutilexpert_10agent_nstep.pkl" # 确认文件名正确
    success_data = load_expert_data_to_agent(agent, expert_data_path)
    
    if not success_data:
        print("did not find expert data ")
    else:
        # 2. 只有数据加载成功，才开始 RL 训练
        print(f"expert data, data amount: {len(agent.expert_replay)}")
        rl_train(env, agent)

    ###########################continue train rl 2 #########################################################
    # train_phase_2(env, agent, total_episodes=300, checkpoint_path="mutilDqfd_final_rl_model.pth")

    ###########################continue train rl 3 #########################################################
    # train_phase_3_recovery(env, agent, total_episodes=300, checkpoint_path="mutilDqfd_Phase2_Final.pth")

    # train_phase4(env, agent, total_episodes=200)
    # train_phase5(env, agent, total_episodes=200)

    # train_phase6(env, agent,500)
    # train_phase7(env, agent)
