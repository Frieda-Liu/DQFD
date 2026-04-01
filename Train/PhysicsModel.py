# PhysicsModel.py

class EVPhysics:
    @staticmethod
    def calculate_step_consumption(distance_m, speed_kmh, weather_factor, congestion_factor=1.0):
        """
        统一能耗计算公式
        :param distance_m: 路段长度（米），通常为 354.0
        :param speed_kmh: 该路段的限速（km/h）
        :param weather_factor: 天气系数（影响通行时间，通常 0.8 - 2.0）
        :param congestion_factor: 拥堵系数（影响实际速度，1.0 为畅通）
        :return: (energy_cost, travel_time)
        """
        # 1. 计算受拥堵影响后的实际速度
        # 拥堵越高，实际速度越低，但不能低于 5km/h (模拟蠕行)
        actual_speed_kmh = max(5.0, speed_kmh / congestion_factor)
        speed_ms = actual_speed_kmh / 3.6
        
        # 2. 基础通行时间 (秒)
        # 时间 = 距离 / 速度
        base_time = distance_m / speed_ms if speed_ms > 0 else 10.0
        
        # 3. 应用天气影响
        # 假设 1.2 是标准天气，超过 1.2 以后时间线性增加
        weather_multiplier = 1.0 + max(0.0, (weather_factor - 1.2) * 0.2)
        travel_time = base_time * weather_multiplier
        
        # 4. 核心耗能模型 (基于你之前的公式)
        # 第一部分：时间相关的基础消耗 (如空调、电子设备，0.003)
        # 第二部分：速度相关的动力消耗 (空气阻力等，系数 0.000005)
        # 注意：这里使用限速 speed_kmh 还是实际速度 actual_speed_kmh 取决于你的建模倾向
        # 通常拥堵时虽然速度慢，但频繁启停会导致能耗并不低，这里我们统一使用实际车速
        energy_cost = ((travel_time * 0.03) + (actual_speed_kmh ** 2) * 0.00005) * congestion_factor
        
        return energy_cost, travel_time

    @staticmethod
    def get_standard_time(distance_m, speed_kmh):
        """
        供专家规划路径时使用的标准时间参考（不考虑随机天气和拥堵）
        """
        speed_ms = speed_kmh / 3.6
        return distance_m / speed_ms if speed_ms > 0 else 10.0