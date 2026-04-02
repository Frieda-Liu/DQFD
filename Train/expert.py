# expert.py 
import numpy as np
import math
import networkx as nx
from PhysicsModel import EVPhysics

class Expert:
    """
    Enhanced Expert Policy using Graph-based Pathfinding (Dijkstra/A*).
    Performance Targets: 95%+ success in simple scenarios, 60%+ in high-congestion/complex scenarios.
    """
    
    def __init__(self):
        # State of Charge (SoC) Thresholds for Decision Making
        self.critical_soc = 30        # Emergency charging trigger
        self.planning_soc = 50        # Proactive charging planning trigger
        self.safe_soc = 80            # Target SoC for safety buffer
        self.min_soc_at_arrival = 15  # Minimum reserved energy upon reaching destination
        

    def get_action(self, env, agent_id):
        """
        Determines the optimal action for a specific agent using Dijkstra-based path planning.
        """
        current_pos = env.agent_positions[agent_id]
        current_soc = env.vehicles[agent_id].soc
        target_pos = env.target_positions[agent_id]
        
        # 1. Compute shortest path to goal and estimate energy consumption
        try:
            # Use Dijkstra's algorithm to find the shortest path based on edge length
            path_to_target = nx.shortest_path(env.graph, current_pos, target_pos, weight='length')
            energy_to_target = self.calculate_path_energy(path_to_target, env)
        except nx.NetworkXNoPath:
            # Fallback to Euclidean estimation if no path exists in the graph
            path_to_target = None
            energy_to_target = math.dist(current_pos, target_pos) * 2.0
        
        # 2. Intelligent Charging Logic
        if current_soc < self.critical_soc and env.charging_stations:
            # Emergency: Prioritize the nearest reachable charging station
            target = self.find_best_emergency_charger(current_pos, current_soc, env)
            if target:
                return self.move_towards_target(agent_id, current_pos, target, env)
        
        elif current_soc < self.planning_soc and energy_to_target > current_soc * 0.7:
            # Planning: Find optimal charger considering detour costs and charging speed (L2 vs L3)
            best_charger = self.find_optimal_charger(current_pos, target_pos, current_soc, env)
            if best_charger:
                return self.move_towards_target(agent_id, current_pos, best_charger, env)
        
        # 3. Standard Navigation to Goal
        return self.move_towards_target(agent_id, current_pos, target_pos, env)
    
    def calculate_path_energy(self, path, env):
        """
        Calculates the total estimated energy consumption for a given path sequence.
        """
        if not path or len(path) < 2:
            return float('inf')
        
        total_energy = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            edge_data = env.graph.get_edge_data(u, v, default={})
           
            speed = edge_data.get('speed', 40.0)

            # Local Density estimation for congestion simulation
            density = 1.0
            if hasattr(env, 'agent_positions'):
                density = sum(1 for p in env.agent_positions if p == v)

            # Utilize the shared Physics Model for consistency with the Environment
            energy, _ = EVPhysics.calculate_step_consumption(
                env.H3_LENGTH_METERS, 
                speed, 
                env.weather_factor,
                congestion_factor = 1.0 + max(0, (density - 1) * 0.5)
            )
            total_energy += energy
        
        return total_energy
    
    def find_best_emergency_charger(self, current_pos, current_soc, env):
        """
        Emergency Mode: Identifies the nearest reachable charging station under low battery constraints.
        """
        best_charger = None
        min_distance = float('inf')
        
        for charger_pos in env.charging_stations.keys():
            try:
                path = nx.shortest_path(env.graph, current_pos, charger_pos, weight='length')
                energy_needed = self.calculate_path_energy(path, env)
                
                # Check feasibility based on current battery level
                if energy_needed < current_soc:
                    distance = len(path) # Use step count as distance proxy
                    if distance < min_distance:
                        min_distance = distance
                        best_charger = charger_pos
            except nx.NetworkXNoPath:
                continue
        
        return best_charger
    
    def find_optimal_charger(self, current_pos, target_pos, current_soc, env):
        """
        Optimization Mode: Selects a charging station based on a weighted score of detour cost and charger level.
        """
        best_charger = None
        best_score = -float('inf')
        
        # Baseline Euclidean distance to goal
        direct_distance = math.dist(current_pos, target_pos)
        
        for charger_pos, charger_type in env.charging_stations.items():
            try:
                # Path to charger
                path_to_charger = nx.shortest_path(env.graph, current_pos, charger_pos, weight='length')
                energy_to_charger = self.calculate_path_energy(path_to_charger, env)
                
                if energy_to_charger > current_soc:
                    continue
                
                # Path from charger to goal
                path_from_charger = nx.shortest_path(env.graph, charger_pos, target_pos, weight='length')
                
                total_distance = len(path_to_charger) + len(path_from_charger)
                detour_cost = total_distance - direct_distance
                
                # Reward L3 fast chargers over L2
                charge_speed_bonus = 2.0 if charger_type == "L3" else 1.0
                
                # Heuristic Score: Minimized detour cost + Maximized charging speed
                score = -detour_cost * 0.5 + charge_speed_bonus * 10
                
                # Add urgency weight if battery is critically low
                if current_soc < 20:
                    score += 20
                
                if score > best_score:
                    best_score = score
                    best_charger = charger_pos
                    
            except nx.NetworkXNoPath:
                continue
        
        return best_charger
    
    def move_towards_target(self, agent_id, current_pos, target_pos, env):
        """
        Executes the first step of the planned A* or Dijkstra path.
        """
        try:
            path = nx.shortest_path(env.graph, current_pos, target_pos, weight='length')
            
            if len(path) > 1:
                next_pos = path[1]
                
                # Map coordinate delta to discrete environment actions
                dx = next_pos[0] - current_pos[0]
                dy = next_pos[1] - current_pos[1]
                
                for i, (di, dj) in enumerate(env.directions):
                    if (di, dj) == (dx, dy):
                        return i
        except nx.NetworkXNoPath:
            pass
        
        # Fallback to local heuristic movement if graph-based pathfinding fails
        return self.heuristic_move(agent_id, current_pos, target_pos, env)
    
    def heuristic_move(self, agent_id, current_pos, target_pos, env):
        """
        Heuristic Fallback: Moves greedily towards the target based on vector magnitude and road priority.
        """
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        
        magnitude = math.sqrt(dx**2 + dy**2)
        if magnitude > 0:
            dx /= magnitude
            dy /= magnitude
        
        best_action = 0
        best_score = -float('inf')
        
        for i, (di, dj) in enumerate(env.directions):
            next_pos = (current_pos[0] + di, current_pos[1] + dj)
            
            if next_pos not in env.graph:
                continue
            
            # Directional alignment (Cosine similarity proxy)
            direction_score = dx * di + dy * dj
            
            # Favor main roads for better routing reliability
            road_bonus = 0.5 if next_pos in env.london_main_roads else 0
            
            # Avoid oscillation/looping by checking agent trajectory history
            agent_history = list(env.visited_nodes[agent_id])
            visit_penalty = -0.3 if next_pos in agent_history[-10:] else 0
            
            total_score = direction_score + road_bonus + visit_penalty
            
            if total_score > best_score:
                best_score = total_score
                best_action = i
        
        return best_action