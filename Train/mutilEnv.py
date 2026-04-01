import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
import math
import random
import pickle
import torch
from PhysicsModel import EVPhysics

# ================= 车辆物理常数 =================
CHARGER_LEVELS = {"L2": 15.0, "L3": 50.0} # 每步充电量 (%)
# H3 Res 9 对应分辨率，每边长约 174m，中心间距约 354m

class EVVehicle:
    def __init__(self, battery_capacity=100.0):
        self.max_capacity = battery_capacity
        self.soc = battery_capacity
        self.is_active = True
        self.is_dead = False
        self.goal =False
        self.finish_status = "running"

    def consume(self, cost):
        # print("soc",self.soc,"cost:",cost)
        self.soc = max(0.0, self.soc - cost)
        if self.soc <= 0:
            self.is_dead = True
            self.dead()

    def reach_goal(self):
        self.is_active = False
        self.goal = True
        self.finish_status = "success"

    def crash(self, reason="out_of_road"):
        self.is_active = False
        self.finish_status = reason
    
    def dead(self, reason="out_of_battery"):
        self.is_active = False
        self.finish_status = reason

class HexTrafficEnv(gym.Env):
    def __init__(self, radius=120, num_agents = 5):
        super().__init__()
        #load london map
        try:
            with open("london_data_improved.pkl", "rb") as f:
                map_data = pickle.load(f)
        except FileNotFoundError:
            print("cannot found map data")

        self.london_main_roads = map_data["road_cells"]
        self.speed_map = map_data["speed_map"]
        self.traffic_signals = map_data["traffic_signals"]
        self.charging_stations = map_data.get("chargers", {})
        self.visited_nodes = []
        self.maxsteps = 300
        self.num_agents = num_agents
        self.soc_threshold = 20.0
        
        # 2. 环境配置
        self.radius = radius
        self.H3_LENGTH_METERS = 354.0
        self.directions = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
        
        # 3. action and observation space
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=np.array([-1.0]*4 + [0.0, 0.5] + [-1.0]*2 + [0.0]*6 + [0.0]*6, dtype=np.float32),
            high=np.array([1.0]*4 + [1.0, 2.0] + [1.0]*2 + [1.0]*6 + [1.0]*6, dtype=np.float32),
            dtype=np.float32
        )

        # 4. create graph 
        self.graph = self._create_hex_graph()
        #remove charger station that is not on the graph
        self.charging_stations = {k: v for k, v in self.charging_stations.items() if k in self.graph}
        self.reset()

    def _create_hex_graph(self):
        G = nx.Graph()
        G.add_nodes_from(self.london_main_roads)
        
        for node in list(G.nodes):
            for di, dj in self.directions:
                neighbor = (node[0] + di, node[1] + dj)
                if neighbor in G:
                    # get max speed 
                    speed_kmh = self.speed_map.get(neighbor, 40.0)
                    if speed_kmh <= 0 or speed_kmh > 120:
                        speed_kmh = 40.0 

                    _, b_time = EVPhysics.calculate_step_consumption(
                        self.H3_LENGTH_METERS, 
                        speed_kmh, 
                        1.2  
                    )
                    
                    # singal penalty
                    #id neighbor is singal, add 5-15 second 
                    signal_penalty = random.uniform(5.0, 15.0) if neighbor in self.traffic_signals else 0.0
                    
                    G.add_edge(node, neighbor, 
                            length=self.H3_LENGTH_METERS,
                            speed=speed_kmh,
                            base_time=b_time + signal_penalty,
                            current_time=b_time + signal_penalty)
        
        # check if the graph is all connected 
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
        
        # check every node have maxium speed 
        for u, v, data in G.edges(data=True):
            if 'speed' not in data or data['speed'] <= 0:
                data['speed'] = 40.0
            if 'current_time' not in data or data['current_time'] <= 0:
                data['current_time'] = 10.0
        return G

    def reset(self, seed=None, options=None, episode_idx=None):
        super().reset(seed=seed)
        self.step_count = 0
        nodes = list(self.graph.nodes)
        
        # vehicels list
        self.vehicles = [EVVehicle(battery_capacity=100.0) for _ in range(self.num_agents)]
        self.agent_positions = [random.choice(nodes) for _ in range(self.num_agents)]
        self.target_positions = [random.choice(nodes) for _ in range(self.num_agents)]
        
        # make sure the start and end is not the same 
        for i in range(self.num_agents):
            while self.target_positions[i] == self.agent_positions[i]:
                self.target_positions[i] = random.choice(nodes)
                
        self.weather_factor = random.uniform(0.8, 2.0)
        self.dones = [False] * self.num_agents 

        self.visited_nodes = [set() for _ in range(self.num_agents)]
        for i in range(self.num_agents):
            self.visited_nodes[i].add(self.agent_positions[i])
        
        return self._get_batch_obs(), {}

    def _get_hex_dist(self,p1, p2):
        dq = p1[0] - p2[0]
        dr = p1[1] - p2[1]
        return (abs(dq) + abs(dr) + abs(dq + dr)) / 2.0

    def _get_single_obs(self, i):
        pos = self.agent_positions[i]
        target = self.target_positions[i]
        v = self.vehicles[i]

        # 1. the nearest charager station
       
        if self.charging_stations and v.soc < self.soc_threshold:
            nearest_charger = self._find_nearest_charger(pos)
            c_dx = np.tanh((nearest_charger[0] - pos[0])  / 10.0) 
            c_dy = np.tanh((nearest_charger[1] - pos[1])  / 10.0)
        else:
            c_dx, c_dy = 0.0, 0.0

        # 2. get all congested level 
        current_densities = {}
        for p in self.agent_positions:
            current_densities[p] = current_densities.get(p, 0) + 1

        mask = []
        density_radar = []
        for di, dj in self.directions:
            nb = (pos[0] + di, pos[1] + dj)
            # mask: if node is a movable road
            mask.append(1.0 if nb in self.graph else 0.0)
            # get how many car in this node to check the congested level 
            d_val = current_densities.get(nb, 0)
            density_radar.append(min(1.0, d_val / 10.0))

        # 4. based factor
        obs_base = np.array([
            #self position
            pos[0] / self.radius, 
            pos[1] / self.radius,
            #target location
            (target[0] - pos[0]) / (self.radius * 2), 
            (target[1] - pos[1]) / (self.radius * 2),
            
            # soc
            v.soc / 100.0, 
            #weather 
            self.weather_factor / 1.2,
            #nearest charager station
            c_dx, 
            c_dy,
        ], dtype=np.float32)

        # 拼接：8(基础) + 6(掩码) + 6(雷达) = 20维
        return np.concatenate([obs_base, np.array(mask), np.array(density_radar)], axis=0)

    def _get_batch_obs(self):
        """return all agent observation"""
        obs_list = [self._get_single_obs(i) for i in range(self.num_agents)]
        # print(obs_list)
        return np.array(obs_list, dtype=np.float32)

    def _get_obs(self):
        return self._get_batch_obs()

    def step(self, actions):
        """
        actions: agent list
        return: all agent observation, reward list, tunicated list,reached max step info list
        """
        node_density = {}
        for pos in self.agent_positions:
            node_density[pos] = node_density.get(pos, 0) + 1
        rewards = []
        terminateds = []
        truncateds = []
        infos = [{} for _ in range(self.num_agents)]

        # update agent state
        for i in range(self.num_agents):
            # if done, move to next agent
            if self.dones[i]:
                rewards.append(0.0)
                terminateds.append(True)
                truncateds.append(False)
                continue
            # 2. next position
            di, dj = self.directions[actions[i]]
            target_next_pos = (self.agent_positions[i][0] + di, self.agent_positions[i][1] + dj)

            # 3. get congestion level
            next_node_density = node_density.get(target_next_pos, 0)

            # get single agent reward 
            res = self._step_single_agent(i, actions[i],next_node_density)
            # print(res)
            
            rewards.append(res["reward"])
            terminateds.append(res["terminated"])
            truncateds.append(res["truncated"])
            infos[i] = res["info"] # 覆盖预设的空字典
            # update status 
            if res["terminated"] or res["truncated"]:
                self.dones[i] = True

        self.step_count += 1
        if self.step_count >= self.maxsteps:
            for i in range(self.num_agents):
                if not self.dones[i]:
                    rewards[i] -= 200.0 
                    self.dones[i] = True 
                    infos[i]["reason"] = "timeout_penalty"
        
        # finiahed when all agents are done 
        all_done = all(self.dones)
        
        return self._get_obs(), rewards, all_done, self.step_count >= self.maxsteps, infos

    def _step_single_agent(self, i, action,density):
        """handle single agent 's movement and reward"""
        # --- current position ---
        pos = self.agent_positions[i]
        target = self.target_positions[i]
        v = self.vehicles[i]
        if not v.is_active:
            return {"reward": 0.0, "terminated": True, "info": {"reason": v.finish_status}}
      
        reward = -3.0
        info = {"agent":i,"success": False, "reason": "moving","current ":pos,"soc":v.soc}
        
        terminated = False
        truncated = False
        # 1. move
        di, dj = self.directions[action]
        next_pos = (pos[0] + di, pos[1] + dj)

        # 2. out of road 
        if next_pos not in self.graph:
            v.consume(0.2 * self.weather_factor)
            v.crash("out_of_road")
            reward = -10.0
            # print("out of road ",reward)
            return {
                "reward": reward, 
                "terminated": True, 
                "truncated": False, 
                "info": {"reason": "out_of_road", "success": False} 
            }

        edge_data = self.graph.get_edge_data(pos, next_pos)
        congestion = 1.0 + max(0, (density - 1) * 0.1)
      
        energy_cost, travel_time = EVPhysics.calculate_step_consumption(
            distance_m=self.H3_LENGTH_METERS,
            speed_kmh=edge_data.get('speed', 40.0),
            weather_factor=self.weather_factor,
            congestion_factor=congestion
        )

        v.consume(energy_cost)

        if density > 1:
          # 20% of cars
          dynamic_threshold = max(1, self.num_agents * 0.20)
          if density > dynamic_threshold:
              reward -= (density - dynamic_threshold) * (5.0 / self.num_agents)
            #   print(" density > 1",(density - dynamic_threshold) * (5.0 / self.num_agents))

        # 4. check if teh battery is below 35, then find the charage station first
        is_low_battery = v.soc < self.soc_threshold
        ref_pos = self._find_nearest_charger(next_pos) if is_low_battery else target
        
        dist_old = self._get_hex_dist(pos, ref_pos)
        dist_new = self._get_hex_dist(next_pos, ref_pos)
        #if close to the target or charager station get more reward
        #else punished 
        dist_diff = dist_old - dist_new
        reward += dist_diff * 10

        soc = v.soc
        if soc <= self.soc_threshold:
            anxiety_penalty = -0.1 * ((self.soc_threshold - soc) ** 2)
            reward += anxiety_penalty
            
        #revisited
        if next_pos in self.visited_nodes[i]:
          reward -= 5.0

        # 5. charge get extra reward
        charged = False
        if next_pos in self.charging_stations:
            # print("charging....")
            # print("is_low_battery",is_low_battery)
            if is_low_battery:
                reward += 50
            elif v.soc >= 85:
                reward+=-1.0
            else:
                reward += 0
            charge_rate = CHARGER_LEVELS.get(self.charging_stations[next_pos], 15.0)
            v.soc = min(100.0, v.soc + charge_rate)
            charged = True

        # 6. update location
        self.agent_positions[i] = next_pos
        self.visited_nodes[i].add(next_pos)

        # 7. arrived at target
        dist_to_target = self._get_hex_dist(next_pos, target)
        if dist_to_target < 1.1 or next_pos == target:
            reward += 500.0
            v.reach_goal()
            return {"reward": reward, "terminated": True,"truncated": False,  "info": {"reason": "success"}}
        if v.is_dead:
            reward -= 100.0
            return {
                "reward": reward, 
                "terminated": True, 
                "truncated": False, 
                "info": {"reason": "out_of_battery", "success": False}
            }

        return {
            "reward": reward, 
            "terminated": False, 
            "truncated": False, 
            "info": {"reason": "running", "success": False, "soc": v.soc}
        }

    def _find_nearest_charger(self, pos):
        return min(self.charging_stations.keys(), 
               key=lambda c: self._get_hex_dist(pos, c))