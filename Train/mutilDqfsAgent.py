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
        print(f"Using device: {self.device}")
        
        # Neural Network Architecture
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
        
        # Training Hyperparameters
        self.batch_size = 128
        self.gamma = 0.99
        self.epsilon = 0.5
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.target_update = 100

        self.expert = Expert()
        
        # Expert Demonstration Buffer
        self.expert_replay = deque(maxlen=200000)
        
        self.step_count = 0
    
    def collect_expert_demos(self, env, num_episodes=2000, n_step=10):
        print(f"\nStarting expert data collection...")
        success_count = 0
        
        for ep in range(num_episodes):
            obs, _ = env.reset() 
            # Apply dynamic weather scenarios
            env.weather_factor = 1.0 if ep < num_episodes * 0.3 else 1.3 if ep < num_episodes * 0.7 else 1.6
            
            for u, v, d in env.graph.edges(data=True):
                d['current_time'] = d.get('base_time', 10.0)
                d['speed'] = env.speed_map.get(v, 40.0) 

            # Randomly simulate traffic congestion
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
                    
                    # Store current trajectory step
                    temp_buffers[i].append({
                        's': obs[i].copy(), 
                        'a': actions[i], 
                        'r': rewards[i], 
                        's_next': next_obs[i].copy(), 
                        'd': float(env.dones[i])
                    })
                    
                    # Handle N-step returns or episode termination
                    if len(temp_buffers[i]) >= n_step or env.dones[i]:
                        is_success = v.goal
                        is_running = v.is_active
                        
                        # Calculate n-step cumulative reward
                        R_n = sum([(self.gamma ** idx) * step['r'] for idx, step in enumerate(temp_buffers[i])])
                        
                        first_step = temp_buffers[i][0]
                        last_step = temp_buffers[i][-1]
                        
                        # Only record demonstrations where the agent is successful or still active
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
                            if is_success: 
                                success_count += 1

                obs = next_obs
                if env.step_count >= env.maxsteps: break
                
            if (ep + 1) % 50 == 0:
                print(f"Progress: {ep+1}/{num_episodes} | Buffer Size: {len(self.expert_replay)} | Successful Episodes: {success_count}")
            
    def pretrain_with_expert(self, epochs=200):
        dataset_size = len(self.expert_replay)
        expert_list = list(self.expert_replay)
        
        for epoch in range(epochs):
            random.shuffle(expert_list) # Reshuffle every epoch to ensure coverage
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
                
                # TD Targets calculation (1-step & n-step)
                with torch.no_grad():
                    next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
                    target_q = rewards + (1 - dones) * self.gamma * next_q
                    
                    next_q_n = self.target_net(s_n_next).max(1)[0].unsqueeze(1)
                    target_q_n = r_n + (1 - d_n) * (self.gamma ** 10) * next_q_n

                # 1. 1-step MSE Loss
                loss_dq = nn.MSELoss()(current_q, target_q)
                # 2. N-step MSE Loss
                loss_n = nn.MSELoss()(current_q, target_q_n)
                # 3. Large Margin Supervised Loss (JE) - Encourages expert action superiority
                l = torch.ones_like(q_values) * 0.8
                l.scatter_(1, actions, 0)
                max_q_with_margin = (q_values + l).max(1)[0]
                loss_je = (max_q_with_margin - current_q.squeeze()).mean()
                
                # 4. L2 Regularization
                l2_reg = sum(p.pow(2.0).sum() for p in self.policy_net.parameters())

                # Total Combined DQfD Pre-training Loss
                batch_loss = loss_dq + 1.0*loss_n + 1.0*loss_je + 1e-5*l2_reg
                
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
                
                epoch_loss += batch_loss.item()
                correct_actions += (q_values.argmax(1).unsqueeze(1) == actions).sum().item()

            accuracy = correct_actions / dataset_size
            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / (dataset_size / self.batch_size)
                print(f"Epoch [{epoch+1}/{epochs}] | Avg Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2%}")
                self.target_net.load_state_dict(self.policy_net.state_dict())

    def select_action(self, state, env, agent_id=0, training=True, expert_weight=0.5):
        # 1) Expert Induction: Guide agent using expert policy
        if training and random.random() < expert_weight:
            return self.expert.get_action(env, agent_id=agent_id)

        # 2) Epsilon-Greedy Exploration 
        if training and random.random() < self.epsilon:
            mask = state[8:14] # Assuming 8:14 corresponds to move direction availability
            legal_actions = [i for i, m in enumerate(mask) if m >= 0.5]
            if legal_actions:
                return random.choice(legal_actions)
            return random.randint(0, 5)
        
        # 3) Network Prediction (Exploitation)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.policy_net(state_tensor)[0].cpu().numpy()

        # Apply action mask to prevent illegal moves (off-road)
        mask = state[8:14] 
        q[mask < 0.5] = -1e9
        return int(np.argmax(q))
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def update_hybrid(self):
        """
        Hybrid update utilizing both self-generated experience and expert demonstrations.
        """
        if len(self.memory) < self.batch_size:
            return 0.0

        # 1. Sample from Expert Replay Buffer
        expert_batch = random.sample(self.expert_replay, self.batch_size // 2)
        s_e, a_e, r_e, sn_e, d_e, rn_e, snn_e, dn_e = zip(*expert_batch)

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
        
        # Supervised Large Margin Loss (JE)
        margin = 0.8
        l = torch.ones_like(q_e) * margin
        l.scatter_(1, a_e, 0)
        max_q_with_margin = (q_e + l).max(1)[0]
        loss_je = (max_q_with_margin - current_q_e.squeeze()).mean()

        # Weight Regularization
        l2_reg = torch.tensor(0.).to(self.device)
        for param in self.policy_net.parameters():
            l2_reg += torch.norm(param)

        # 3. Sample from Agent's Own Experience Memory
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
        
        # Combine all losses with dynamic JE weighting
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
        print("Model loaded successfully.")
    except:
        print("Failed to load model. Starting from scratch.")

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

        # --- Dynamic Expert Weight Adjustment ---
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
                # Filter out extreme negative rewards to focus on quality transitions
                if rewards[i] > -50: 
                    agent_done = env.dones[i]
                    agent.store_transition(obs[i], actions[i], rewards[i], next_obs[i], float(agent_done))
            
            # Hybrid training step
            loss = agent.update_hybrid() 
            if loss is not None:
                ep_losses.append(loss)
            
            obs = next_obs
            done = all_done or truncated
        
        # Logging reward and loss history
        history_rewards.append(ep_reward)
        if len(ep_losses) > 0:
            history_losses.append(sum(ep_losses) / len(ep_losses))
        else:
            history_losses.append(0.0)

        if (ep + 1) % 10 == 0:
            avg_reward = sum(history_rewards[-10:]) / 10
            avg_loss = sum(history_losses[-10:]) / 10
            print(f"Episode {ep+1}/{total_episodes} | Avg Reward: {avg_reward:.2f} | Avg Loss: {avg_loss:.4f} | Expert_W: {cur_expert_w:.2f} | Eps: {agent.epsilon:.3f}")

    torch.save(agent.policy_net.state_dict(), "mutilDqfd_final_rl_model.pth")
    print("Final RL model saved as mutilDqfd_final_rl_model.pth")
    return history_rewards, history_losses

def run_collection_and_save(agent, env, filename="mutilexpert_data_nstep.pkl", num_episodes=100):
    """
    Handles the data collection phase using the expert policy.
    """
    if os.path.exists(filename):
        print(f"Warning: {filename} already exists.")
        cont = input("Recollect data? (y/n): ")
        if cont.lower() != 'y':
            return
    
    agent.collect_expert_demos(env, num_episodes=num_episodes)

    with open(filename, 'wb') as f:
        pickle.dump(agent.expert_replay, f)
    
    print(f"Expert data saved at {filename}")
    print(f"Total entries: {len(agent.expert_replay)}")

def load_expert_data_to_agent(agent, filename="mutilexpert_data_nstep.pkl"):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            agent.expert_replay = pickle.load(f)
        print(f"Successfully loaded expert data. Length: {len(agent.expert_replay)}")
        return True
    else:
        print(f"File not found: {filename}")
        return False

if __name__ == "__main__":
    # 1. Environment and Agent Initialization
    env = HexTrafficEnv(radius=120, num_agents=10)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = ExpertDQN(state_dim, action_dim) 

    ###################### PHASE 1: COLLECT EXPERT DATA ##################################
    # run_collection_and_save(agent, env, filename="mutilexpert_10agent_nstep.pkl", num_episodes=1000)

    ###################### PHASE 2: PRE-TRAINING ########################################
    # success = load_expert_data_to_agent(agent, "mutilexpert_10agent_nstep.pkl")
    # if success:
    #     print("Starting Pre-training phase...")
    #     agent.pretrain_with_expert(epochs=250)
    #     torch.save(agent.policy_net.state_dict(), "mutildqfd_pretrained_10agent_model.pth")
    #     print("Pre-training model saved.")

    ###################### PHASE 3: RL TRAINING ##########################################
    expert_data_path = "mutilexpert_10agent_nstep.pkl" 
    success_data = load_expert_data_to_agent(agent, expert_data_path)
    
    if not success_data:
        print("Expert data not found. RL training aborted.")
    else:
        print(f"Expert data confirmed. Total records: {len(agent.expert_replay)}")
        rl_train(env, agent)