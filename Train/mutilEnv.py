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

# Charging speeds based on charger levels (kWh per step/simulation unit)
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
        """Deducts energy and checks for battery depletion."""
        self.soc = max(0.0, self.soc - cost)
        if self.soc <= 0:
            self.is_dead = True
            self.is_active = False
            self.finish_status = "out_of_battery"

    def reach_goal(self):
        """Sets agent status to successful arrival."""
        self.is_active = False
        self.goal = True
        self.finish_status = "success"

    def crash(self, reason="out_of_road"):
        """Handles off-road or collision termination."""
        self.is_active = False
        self.finish_status = reason

class HexTrafficEnv(gym.Env):
    def __init__(self, radius=120, num_agents=5):
        super().__init__()
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(current_dir)
    
        # Load improved London geographic data
        pkl_path = os.path.join(root_dir, "MAP", "london_data_improved.pkl")
        if not os.path.exists(pkl_path):
            pkl_path = os.path.join(current_dir, "london_data_improved.pkl")

        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Map data not found at: {pkl_path}")

        with open(pkl_path, "rb") as f:
            map_data = pickle.load(f)
       
        self.london_main_roads = map_data.get("road_cells", map_data.get("road_nodes", []))
        if not self.london_main_roads:
            raise KeyError("Critical error: Could not find road_cells or road_nodes in map data.")

        self.speed_map = map_data.get("speed_map", {})
        self.traffic_signals = map_data.get("traffic_signals", {})
        self.charging_stations = map_data.get("chargers", {})
        
        self.maxsteps = 300
        self.num_agents = num_agents
        self.soc_threshold = 20.0 # Battery level trigger for charging behavior
        self.radius = radius
        self.H3_LENGTH_METERS = 354.0 # Distance between centers of H3 resolution 8 cells
        self.directions = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
        
        # Action Space: 6 discrete directions in a hexagonal grid
        self.action_space = spaces.Discrete(6)
        
        # Observation Space (20 dimensions):
        # [0-1] Normalized Pos, [2-3] Dist to Target, [4] SoC, [5] Weather, [6-7] Nearest Charger, 
        # [8-13] Adjacency Mask, [14-19] Traffic Density Radar
        self.observation_space = spaces.Box(
            low=np.array([-1.0]*4 + [0.0, 0.5] + [-1.0]*2 + [0.0]*6 + [0.0]*6, dtype=np.float32),
            high=np.array([1.0]*4 + [1.0, 2.0] + [1.0]*2 + [1.0]*6 + [1.0]*6, dtype=np.float32),
            dtype=np.float32
        )

        self.graph = self._create_hex_graph()
        # Filter out charging stations that are not reachable in the filtered graph
        self.charging_stations = {k: v for k, v in self.charging_stations.items() if k in self.graph}
        self.reset()

    def _create_hex_graph(self):
        """Constructs a NetworkX graph based on H3 hexagonal adjacency."""
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
        
        # Ensure graph connectivity for valid pathfinding
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
        self.trajectories = [[pos] for pos in self.agent_positions]
        return self._get_batch_obs(), {}

    def _get_hex_dist(self, p1, p2):
        """Calculates Axial/Manhattan distance on a hexagonal grid."""
        dq, dr = p1[0] - p2[0], p1[1] - p2[1]
        return (abs(dq) + abs(dr) + abs(dq + dr)) / 2.0

    def _get_single_obs(self, i):
        """Constructs the local observation for agent i."""
        pos, target, v = self.agent_positions[i], self.target_positions[i], self.vehicles[i]
        
        # Gated Attention: Perceive chargers only when SoC is low
        if self.charging_stations and v.soc < self.soc_threshold:
            nearest_charger = self._find_nearest_charger(pos)
            c_dx = np.tanh((nearest_charger[0] - pos[0]) / 10.0) 
            c_dy = np.tanh((nearest_charger[1] - pos[1]) / 10.0)
        else:
            c_dx, c_dy = 0.0, 0.0

        # Congestion Radar: Detect density in immediate hexagonal neighbors
        densities = {p: sum(1 for pos in self.agent_positions if pos == p) for p in set(self.agent_positions)}
        mask, radar = [], []
        for di, dj in self.directions:
            nb = (pos[0] + di, pos[1] + dj)
            mask.append(1.0 if nb in self.graph else 0.0) # Mask for valid road cells
            radar.append(min(1.0, densities.get(nb, 0) / 10.0)) # Local congestion level

        obs_base = np.array([
            pos[0]/self.radius, pos[1]/self.radius,
            (target[0]-pos[0])/(self.radius*2), (target[1]-pos[1])/(self.radius*2),
            v.soc/100.0, self.weather_factor/1.2, c_dx, c_dy
        ], dtype=np.float32)

        return np.concatenate([obs_base, mask, radar])

    def _get_batch_obs(self):
        return np.array([self._get_single_obs(i) for i in range(self.num_agents)], dtype=np.float32)

    def step(self, actions):
        """Executes a joint action for all agents."""
        # Ensure actions list consistency
        if not isinstance(actions, list):
            actions = [actions]
        if len(actions) < self.num_agents:
            actions = list(actions) + [0] * (self.num_agents - len(actions))
        elif len(actions) > self.num_agents:
            actions = list(actions)[:self.num_agents]

        # Pre-calculate current density for congestion penalties
        node_density = {p: sum(1 for pos in self.agent_positions if pos == p) for p in set(self.agent_positions)}
        
        rewards, terminateds, truncateds, infos = [], [], [], [{} for _ in range(self.num_agents)]

        for i in range(self.num_agents):
            if self.dones[i]:
                # Preserve state for inactive agents
                rewards.append(0.0)
                terminateds.append(True)
                truncateds.append(False)
                infos[i] = {"reason": self.vehicles[i].finish_status}
                continue
            
            act_idx = actions[i]
            target_next = (self.agent_positions[i][0] + self.directions[act_idx][0], 
                           self.agent_positions[i][1] + self.directions[act_idx][1])
            
            # Process individual agent step
            res = self._step_single_agent(i, act_idx, node_density.get(target_next, 0))
            
            rewards.append(res["reward"])
            terminateds.append(res["terminated"])
            truncateds.append(res["truncated"])
            infos[i] = res["info"]
            
            if res["terminated"] or res["truncated"]:
                self.dones[i] = True

        # Global step increment and timeout handling
        self.step_count += 1
        if self.step_count >= self.maxsteps:
            for i in range(self.num_agents):
                if not self.dones[i]:
                    rewards[i] -= 200.0  # Significant penalty for timeout
                    self.dones[i] = True
                    infos[i]["reason"] = "timeout_penalty"
        
        return self._get_obs(), rewards, all(self.dones), self.step_count >= self.maxsteps, infos

    def _step_single_agent(self, i, action, density):
        """Processes movement, physics, and reward logic for a single agent."""
        pos, target, v = self.agent_positions[i], self.target_positions[i], self.vehicles[i]
        if not v.is_active: return {"reward": 0.0, "terminated": True, "info": {"reason": v.finish_status}}
  
        reward = -3.0 # Step penalty to encourage efficiency
        di, dj = self.directions[action]
        next_pos = (pos[0] + di, pos[1] + dj)
        self.trajectories[i].append(next_pos)

        # Off-road check
        if next_pos not in self.graph:
            v.crash("out_of_road")
            return {"reward": -10.0, "terminated": True, "truncated": False, "info": {"reason": "out_of_road"}}

        # EV Physics: Energy consumption calculation
        edge_data = self.graph.get_edge_data(pos, next_pos)
        congestion = 1.0 + max(0, (density - 1) * 0.1)
        energy_cost, _ = EVPhysics.calculate_step_consumption(
            self.H3_LENGTH_METERS, edge_data.get('speed', 40.0), self.weather_factor, congestion
        )
        v.consume(energy_cost)

        # Multi-Agent Congestion Penalty
        if density > max(1, self.num_agents * 0.20):
            reward -= (density - (self.num_agents * 0.20)) * (5.0 / self.num_agents)

        # Dynamic Target Selection: Navigate to charger if battery is low, otherwise target
        is_low_battery = v.soc < self.soc_threshold
        ref_pos = self._find_nearest_charger(next_pos) if is_low_battery else target
        
        # Distance-based reward (Potential field logic)
        dist_diff = self._get_hex_dist(pos, ref_pos) - self._get_hex_dist(next_pos, ref_pos)
        reward += dist_diff * 10

        # Battery Anxiety and Looping penalties
        if v.soc <= self.soc_threshold:
            reward -= 0.1 * ((self.soc_threshold - v.soc) ** 2)
        if next_pos in self.visited_nodes[i] and not is_low_battery:
            reward -= 5.0 # Penalty for repeated visits to prevent oscillation

        # Charging Logic
        if next_pos in self.charging_stations:
            if is_low_battery:
                reward += 50.0
                v.soc = min(100.0, v.soc + CHARGER_LEVELS.get(self.charging_stations[next_pos], 15.0))
            elif v.soc >= 85: 
                reward -= 2.0 # Penalty for unnecessary charging

        self.agent_positions[i] = next_pos
        self.visited_nodes[i].add(next_pos)

        # Termination conditions
        if self._get_hex_dist(next_pos, target) < 1.1:
            v.reach_goal()
            return {"reward": 500.0, "terminated": True, "truncated": False, "info": {"reason": "success", "soc": v.soc}}
        if v.is_dead:
            return {"reward": -100.0, "terminated": True, "truncated": False, "info": {"reason": "out_of_battery", "soc": v.soc}}

        return {"reward": reward, "terminated": False, "truncated": False, "info": {"reason": "running", "soc": v.soc}}

    def _find_nearest_charger(self, pos):
        """Returns coordinates of the nearest charging station."""
        return min(self.charging_stations.keys(), key=lambda c: self._get_hex_dist(pos, c))