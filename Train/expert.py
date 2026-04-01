#expert.py 
import numpy as np
import math
import networkx as nx
from PhysicsModel import EVPhysics


class Expert:
    """增强版专家策略 - 目标：简单场景95%+，困难场景60%+"""
    
    def __init__(self):
        self.critical_soc = 30      # 紧急充电阈值
        self.planning_soc = 50      # 规划充电阈值（降低）
        self.safe_soc = 80          # 安全电量（降低）
        self.min_soc_at_arrival = 15  # 到达目标的最小保留电量（降低）
        
    # def get_action(self, env):
    #     """获取专家动作 - 使用A*路径规划"""
    #     current_pos = env.agent_pos
    #     current_soc = env.vehicle.soc
    #     target_pos = env.target_pos
        
    #     # 1. 使用Dijkstra计算最短路径和能耗
    #     try:
    #         # 计算到目标的最短路径
    #         path_to_target = nx.shortest_path(env.graph, current_pos, target_pos, weight='length')
    #         energy_to_target = self.calculate_path_energy(path_to_target, env)
    #     except:
    #         # 如果没路径，直接欧氏距离估算
    #         path_to_target = None
    #         energy_to_target = math.dist(current_pos, target_pos) * 2.0
        
    #     # 2. 智能充电决策
    #     if current_soc < self.critical_soc and env.charging_stations:
    #         # 紧急情况：找最近且可达的充电桩
    #         target = self.find_best_emergency_charger(current_pos, current_soc, env)
    #         if target:
    #             return self.move_towards_target(current_pos, target, env)
        
    #     elif current_soc < self.planning_soc and energy_to_target > current_soc * 0.7:
    #         # 需要充电：找最优充电桩（考虑绕路成本和充电速度）
    #         best_charger = self.find_optimal_charger(current_pos, target_pos, current_soc, env)
    #         if best_charger:
    #             return self.move_towards_target(current_pos, best_charger, env)
        
    #     # 3. 直接去目标
    #     return self.move_towards_target(current_pos, target_pos, env)
    def get_action(self, env,agent_id):
        """获取专家动作 - 使用A*路径规划"""
        current_pos = env.agent_positions[agent_id]
        current_soc = env.vehicles[agent_id].soc
        target_pos = env.target_positions[agent_id]
        
        # 1. 使用Dijkstra计算最短路径和能耗
        try:
            # 计算到目标的最短路径
            path_to_target = nx.shortest_path(env.graph, current_pos, target_pos, weight='length')
            energy_to_target = self.calculate_path_energy(path_to_target, env)
        except:
            # 如果没路径，直接欧氏距离估算
            path_to_target = None
            energy_to_target = math.dist(current_pos, target_pos) * 2.0
        
        # 2. 智能充电决策
        if current_soc < self.critical_soc and env.charging_stations:
            # 紧急情况：找最近且可达的充电桩
            target = self.find_best_emergency_charger(current_pos, current_soc, env)
            if target:
                return self.move_towards_target(agent_id,current_pos, target, env)
        
        elif current_soc < self.planning_soc and energy_to_target > current_soc * 0.7:
            # 需要充电：找最优充电桩（考虑绕路成本和充电速度）
            best_charger = self.find_optimal_charger(current_pos, target_pos, current_soc, env)
            if best_charger:
                return self.move_towards_target(agent_id,current_pos, best_charger, env)
        
        # 3. 直接去目标
        return self.move_towards_target(agent_id,current_pos, target_pos, env)
    
    def calculate_path_energy(self, path, env):
        """计算完整路径的能耗"""
        if not path or len(path) < 2:
            return float('inf')
        
        total_energy = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            edge_data = env.graph.get_edge_data(u, v, {})
           
            # 速度
            speed = edge_data.get('speed', 40.0)

            density = 1.0
            if hasattr(env, 'agent_positions'):
                # 简单的密度统计
                density = sum(1 for p in env.agent_positions if p == v)

            energy, _ = EVPhysics.calculate_step_consumption(
                env.H3_LENGTH_METERS, 
                speed, 
                env.weather_factor,
                congestion_factor = 1.0 + max(0, (density - 1) * 0.5) # 同步环境的拥堵逻辑
            )
            
            # 使用环境相同的能耗公式
            # base_time = distance * 2.0
            # travel_time = base_time * env.weather_factor
            # time_consumption = travel_time * 0.005
            # speed_consumption = (speed ** 2) * 0.00001 * distance
            # energy = (time_consumption + speed_consumption) * env.weather_factor
            energy, _ = EVPhysics.calculate_step_consumption(env.H3_LENGTH_METERS, speed, env.weather_factor)
            total_energy += energy
        
        return total_energy
    
    def find_best_emergency_charger(self, current_pos, current_soc, env):
        """紧急情况：找最近且可达的充电桩"""
        best_charger = None
        min_distance = float('inf')
        
        for charger_pos in env.charging_stations.keys():
            try:
                # 检查是否可达
                path = nx.shortest_path(env.graph, current_pos, charger_pos, weight='length')
                
                # 计算所需能量
                energy_needed = self.calculate_path_energy(path, env)
                
                # 如果能到达
                if energy_needed < current_soc:
                    distance = len(path)  # 用步数作为距离
                    if distance < min_distance:
                        min_distance = distance
                        best_charger = charger_pos
            except:
                continue
        
        return best_charger
    
    def find_optimal_charger(self, current_pos, target_pos, current_soc, env):
        """找最优充电桩（考虑绕路成本和充电速度）"""
        best_charger = None
        best_score = -float('inf')
        
        # 到目标的直接距离
        direct_distance = math.dist(current_pos, target_pos)
        
        for charger_pos, charger_type in env.charging_stations.items():
            try:
                # 到充电站的路径
                path_to_charger = nx.shortest_path(env.graph, current_pos, charger_pos, weight='length')
                energy_to_charger = self.calculate_path_energy(path_to_charger, env)
                
                # 如果到不了充电站，跳过
                if energy_to_charger > current_soc:
                    continue
                
                # 从充电站到目标的路径
                path_from_charger = nx.shortest_path(env.graph, charger_pos, target_pos, weight='length')
                energy_from_charger = self.calculate_path_energy(path_from_charger, env)
                
                # 总距离
                total_distance = len(path_to_charger) + len(path_from_charger)
                
                # 绕路成本
                detour_cost = total_distance - direct_distance
                
                # 充电速度奖励（L3快充有更高奖励）
                charge_speed_bonus = 2.0 if charger_type == "L3" else 1.0
                
                # 综合评分：绕路越少越好，充电越快越好
                score = -detour_cost * 0.5 + charge_speed_bonus * 10
                
                # 如果电量非常低，给予额外奖励
                if current_soc < 20:
                    score += 20
                
                if score > best_score:
                    best_score = score
                    best_charger = charger_pos
                    
            except:
                continue
        
        return best_charger
    
    def move_towards_target(self,agent_id, current_pos, target_pos, env):
        """使用A*路径规划的第一步移动"""
        try:
            # 计算最短路径
            path = nx.shortest_path(env.graph, current_pos, target_pos, weight='length')
            
            if len(path) > 1:
                next_pos = path[1]
                
                # 将位置差转换为动作
                dx = next_pos[0] - current_pos[0]
                dy = next_pos[1] - current_pos[1]
                
                for i, (di, dj) in enumerate(env.directions):
                    if (di, dj) == (dx, dy):
                        return i
        except:
            pass
        
        # 回退：启发式移动
        return self.heuristic_move(agent_id,current_pos, target_pos, env)
    
    def heuristic_move(self, agent_id, current_pos, target_pos, env):
        """启发式移动（当路径规划失败时）"""
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
            
            # 方向匹配度
            direction_score = dx * di + dy * dj
            
            # 道路类型奖励
            road_bonus = 0.5 if next_pos in env.london_main_roads else 0
            
            # 避免重复访问
            agent_history = env.visited_nodes[agent_id]
            history_list = list(agent_history)
            visit_penalty = -0.3 if next_pos in history_list[-10:] else 0
            total_score = direction_score + road_bonus + visit_penalty
            
            if total_score > best_score:
                best_score = total_score
                best_action = i
        
        return best_action