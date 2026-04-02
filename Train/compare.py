
import numpy as np
import pandas as pd
import torch
import copy
from tqdm import tqdm  
from mutilEnv import HexTrafficEnv
from mutilDqfsAgent import ExpertDQN

# ---  Dijkstra  ---
def run_dijkstra_episode(env):
  
    obs, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        actions = []
        for i, v in enumerate(env.vehicles):
            if v.finish_status is not None:
                actions.append(0)
                continue
            neighbors = env.get_neighbors(v.current_pos)
            best_action = 0
            min_dist = float('inf')
            
            for action_idx, move in enumerate(env.action_space_map):
                next_pos = (v.current_pos[0] + move[0], v.current_pos[1] + move[1])
                dist = env.get_distance(next_pos, v.goal_pos)
                if dist < min_dist:
                    min_dist = dist
                    best_action = action_idx
            actions.append(best_action)
            
        obs, rewards, all_done, truncated, _ = env.step(actions)
        done = all_done or truncated
    

    success_count = sum([1 for v in env.vehicles if v.goal])
    return success_count, env.vehicles[0].finish_status


def run_dqfd_episode(env, agent, expert_w=0.1):
    obs, _ = env.reset()
    done = False
    
    while not done:
        actions = [int(agent.select_action(o, env, i, True, expert_w)) for i, o in enumerate(obs)]
        obs, rewards, all_done, truncated, _ = env.step(actions)
        done = all_done or truncated
        
    success_count = sum([1 for v in env.vehicles if v.goal])
    return success_count, env.vehicles[0].finish_status


def main_benchmark():

    NUM_TESTS = 100
    WEATHER_FACTORS = [1.0, 1.5, 2.0]
    MODEL_PATH = "models/temp_soup_0.9559.pth" 

    agent = ExpertDQN(state_dim=20, action_dim=6)
    agent.policy_net.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    
    final_results = []

    print(f"Starting Benchmark: DQfD vs Dijkstra ({NUM_TESTS} tests per weather condition)")

    for w in WEATHER_FACTORS:
        dqfd_successes = 0
        dijkstra_successes = 0
        
        for i in tqdm(range(NUM_TESTS), desc=f"Weather {w}x"):
            
            seed = np.random.randint(0, 10000)
            
            # --- Test Dijkstra ---
            np.random.seed(seed)
            env_dijkstra = HexTrafficEnv(num_agents=1)
            env_dijkstra.weather_factor = w
            d_succ, d_status = run_dijkstra_episode(env_dijkstra)
            dijkstra_successes += d_succ
            
            # --- Test DQfD ---
            np.random.seed(seed)
            env_dqfd = HexTrafficEnv(num_agents=1)
            env_dqfd.weather_factor = w
            q_succ, q_status = run_dqfd_episode(env_dqfd, agent)
            dqfd_successes += q_succ
            
        final_results.append({
            "Weather": w,
            "Dijkstra_ASR": dijkstra_successes / NUM_TESTS,
            "DQfD_ASR": dqfd_successes / NUM_TESTS
        })


    df = pd.DataFrame(final_results)
    print("\nFinal Results:")
    print(df)
    df.to_csv("benchmark_results.csv", index=False)
    print("\n Results saved to benchmark_results.csv")

if __name__ == "__main__":
    main_benchmark()