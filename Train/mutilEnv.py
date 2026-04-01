import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
import math
import random
import pickle
import os
import torch
from PhysicsModel import EVPhysics

# ================= 车辆物理常数 =================
CHARGER_LEVELS = {"L2": 15.0, "L3": 50.0} 

class EVVehicle:
    def __init__(self, battery_capacity=100.0):
        self.max_capacity = battery_capacity
        self.soc = battery_capacity
        self.is_active = True
        self.is_dead = False
        self.goal = False
        self.finish_status = "running"

    def consume(self, cost):
        self.soc = max(0.0, self.soc - cost)
        if self.soc <= 0:
            self.is_dead = True
            self.is_active = False
            self.finish_status = "out_of_battery"

    def reach_goal(self):
        self.is_active = False
        self.goal = True
        self.finish_status = "success"

    def crash(self, reason="out_of_road"):
        self.is_active = False
        self.finish_status = reason

class HexTrafficEnv(gym.Env):
    def __init__(self, radius=120, num_agents=5):
        super().__init__()
        
        # --- 1. 自动路径处理 ---
        # 无论在本地还是 Streamlit 云端，都自动寻找同级目录下的地图文件
        base_path = os.path.dirname(os.path.abspath(__file__))
        pkl_path = os.path.join(base_path, "london_data_improved.pkl")
        
        try:
            with open(pkl_path, "rb") as f:
                map_data = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"无法找到地图文件: {pkl_path}。请确保文件已上传至正确目录。")

        # --- 2. 解决 Key 兼容性问题 ---
        # 自动匹配 road_cells 或 road_nodes
        self.london_main_roads = map_data.get("road_cells", map_data.get("road_nodes", []))
        if not self.london_main_roads:
            raise KeyError("地图文件中找不到路网数据 (road_cells/road_nodes)")

        self.speed_map = map_data.get("speed_map", {})
        self.traffic_signals = map_data.get("traffic_signals", {})
        self.charging_stations = map_data.get("chargers", {})
        
        self.maxsteps = 300
        self.num_agents = num_agents
        self.soc_threshold = 20.0
        self.radius = radius
        self.H3_LENGTH_METERS = 354.0
        self.directions = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
        
        # Action/Observation Space (20维)
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=np.array([-1.0]*4 + [0.0, 0.5] + [-1.0]*2 + [0.0]*6 + [0.0]*6, dtype=np.float32),
            high=np.array([1.0]*4 + [1.0, 2.0] + [1.0]*2 + [1.0]*6 + [1.0]*6, dtype=np.float32),
            dtype=np.float32
        )

        self.graph = self._create_hex_graph()
        # 过滤不在图中的充电桩
        self.charging_stations = {k: v for k, v in self.charging_stations.items() if k in self.graph}
        self.reset()

    def _create_hex_graph(self):
        G = nx.Graph()
        G.add_nodes_from(self.london_main_roads)
        
        for node in list(G.nodes):
            for di, dj in self.directions:
                neighbor = (node[0] + di, node[1] + dj)
                if neighbor in G:
                    speed_kmh = self.speed_map.get(neighbor, 40.0)
                    if speed_kmh <= 0 or speed_kmh > 120: speed_kmh = 40.0 

                    _, b_time = EVPhysics.calculate_step_consumption(self.H3_LENGTH_METERS, speed_kmh, 1.2)
                    signal_penalty = random.uniform(5.0, 15.0) if neighbor in self.traffic_signals else 0.0
                    
                    G.add_edge(node, neighbor, 
                            length=self.H3_LENGTH_METERS, speed=speed_kmh,
                            current_time=b_time + signal_penalty)
        
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
        return G

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        nodes = list(self.graph.nodes)
        
        self.vehicles = [EVVehicle(battery_capacity=100.0) for _ in range(self.num_agents)]
        self.agent_positions = [random.choice(nodes) for _ in range(self.num_agents)]
        self.target_positions = [random.choice(nodes) for _ in range(self.num_agents)]
        
        for i in range(self.num_agents):
            while self.target_positions[i] == self.agent_positions[i]:
                self.target_positions[i] = random.choice(nodes)
                
        self.weather_factor = random.uniform(0.8, 2.0)
        self.dones = [False] * self.num_agents 
        self.visited_nodes = [set([self.agent_positions[i]]) for i in range(self.num_agents)]
        
        return self._get_batch_obs(), {}

    def _get_hex_dist(self, p1, p2):
        dq, dr = p1[0] - p2[0], p1[1] - p2[1]
        return (abs(dq) + abs(dr) + abs(dq + dr)) / 2.0

    def _get_single_obs(self, i):
        pos, target, v = self.agent_positions[i], self.target_positions[i], self.vehicles[i]
        
        # 补能感知闸门 (Gated Attention)
        if self.charging_stations and v.soc < self.soc_threshold:
            nearest_charger = self._find_nearest_charger(pos)
            c_dx = np.tanh((nearest_charger[0] - pos[0]) / 10.0) 
            c_dy = np.tanh((nearest_charger[1] - pos[1]) / 10.0)
        else:
            c_dx, c_dy = 0.0, 0.0

        # 拥堵感知
        densities = {p: sum(1 for pos in self.agent_positions if pos == p) for p in set(self.agent_positions)}
        mask, radar = [], []
        for di, dj in self.directions:
            nb = (pos[0] + di, pos[1] + dj)
            mask.append(1.0 if nb in self.graph else 0.0)
            radar.append(min(1.0, densities.get(nb, 0) / 10.0))

        obs_base = np.array([
            pos[0]/self.radius, pos[1]/self.radius,
            (target[0]-pos[0])/(self.radius*2), (target[1]-pos[1])/(self.radius*2),
            v.soc/100.0, self.weather_factor/1.2, c_dx, c_dy
        ], dtype=np.float32)

        return np.concatenate([obs_base, mask, radar])

    def _get_batch_obs(self):
        return np.array([self._get_single_obs(i) for i in range(self.num_agents)], dtype=np.float32)

    def _get_obs(self): return self._get_batch_obs()

    def step(self, actions):
        node_density = {p: sum(1 for pos in self.agent_positions if pos == p) for p in set(self.agent_positions)}
        rewards, terminateds, truncateds, infos = [], [], [], [{} for _ in range(self.num_agents)]

        for i in range(self.num_agents):
            if self.dones[i]:
                rewards.append(0.0); terminateds.append(True); truncateds.append(False)
                continue
            
            target_next = (self.agent_positions[i][0] + self.directions[actions[i]][0], 
                           self.agent_positions[i][1] + self.directions[actions[i]][1])
            
            res = self._step_single_agent(i, actions[i], node_density.get(target_next, 0))
            rewards.append(res["reward"])
            terminateds.append(res["terminated"])
            truncateds.append(res["truncated"])
            infos[i] = res["info"]
            if res["terminated"] or res["truncated"]: self.dones[i] = True

        self.step_count += 1
        if self.step_count >= self.maxsteps:
            for i in range(self.num_agents):
                if not self.dones[i]:
                    rewards[i] -= 200.0; self.dones[i] = True
                    infos[i]["reason"] = "timeout_penalty"
        
        return self._get_obs(), rewards, all(self.dones), self.step_count >= self.maxsteps, infos

    def _step_single_agent(self, i, action, density):
        pos, target, v = self.agent_positions[i], self.target_positions[i], self.vehicles[i]
        if not v.is_active: return {"reward": 0.0, "terminated": True, "info": {"reason": v.finish_status}}
      
        reward = -3.0
        di, dj = self.directions[action]
        next_pos = (pos[0] + di, pos[1] + dj)

        # 出界判断
        if next_pos not in self.graph:
            v.crash("out_of_road")
            return {"reward": -10.0, "terminated": True, "truncated": False, "info": {"reason": "out_of_road"}}

        # 物理能耗计算
        edge_data = self.graph.get_edge_data(pos, next_pos)
        congestion = 1.0 + max(0, (density - 1) * 0.1)
        energy_cost, _ = EVPhysics.calculate_step_consumption(self.H3_LENGTH_METERS, edge_data.get('speed', 40.0), self.weather_factor, congestion)
        v.consume(energy_cost)

        # 拥堵惩罚
        if density > max(1, self.num_agents * 0.20):
            reward -= (density - (self.num_agents * 0.20)) * (5.0 / self.num_agents)

        # 动态目标点选择
        is_low_battery = v.soc < self.soc_threshold
        ref_pos = self._find_nearest_charger(next_pos) if is_low_battery else target
        
        dist_diff = self._get_hex_dist(pos, ref_pos) - self._get_hex_dist(next_pos, ref_pos)
        reward += dist_diff * 10

        # 焦虑惩罚与重复惩罚
        if v.soc <= self.soc_threshold:
            reward -= 0.1 * ((self.soc_threshold - v.soc) ** 2)
        if next_pos in self.visited_nodes[i] and not is_low_battery:
            reward -= 5.0

        # 补能逻辑
        if next_pos in self.charging_stations:
            if is_low_battery:
                reward += 50.0
                v.soc = min(100.0, v.soc + CHARGER_LEVELS.get(self.charging_stations[next_pos], 15.0))
            elif v.soc >= 85: reward -= 2.0

        self.agent_positions[i] = next_pos
        self.visited_nodes[i].add(next_pos)

        # 状态检查
        if self._get_hex_dist(next_pos, target) < 1.1:
            v.reach_goal()
            return {"reward": 500.0, "terminated": True, "truncated": False, "info": {"reason": "success", "soc": v.soc}}
        if v.is_dead:
            return {"reward": -100.0, "terminated": True, "truncated": False, "info": {"reason": "out_of_battery", "soc": v.soc}}

        return {"reward": reward, "terminated": False, "truncated": False, "info": {"reason": "running", "soc": v.soc}}

    def _find_nearest_charger(self, pos):
        return min(self.charging_stations.keys(), key=lambda c: self._get_hex_dist(pos, c))