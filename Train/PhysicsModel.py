# PhysicsModel.py

class EVPhysics:
    @staticmethod
    def calculate_step_consumption(distance_m, speed_kmh, weather_factor, congestion_factor=1.0):
        """
        Unified energy consumption and travel time calculation formula.
        
        :param distance_m: Segment length in meters (default: 354.0m for H3 resolution 8).
        :param speed_kmh: Speed limit of the segment (km/h).
        :param weather_factor: Weather impact coefficient (affects travel time, typically 0.8 - 2.0).
        :param congestion_factor: Traffic density factor (affects actual speed, 1.0 represents clear flow).
        :return: A tuple of (energy_cost, travel_time).
        """
        # 1. Calculate actual speed influenced by congestion
        # Higher congestion reduces actual speed, capped at a minimum of 5km/h to simulate crawling traffic.
        actual_speed_kmh = max(5.0, speed_kmh / congestion_factor)
        speed_ms = actual_speed_kmh / 3.6
        
        # 2. Calculate baseline travel time (seconds)
        # Time = Distance / Speed
        base_time = distance_m / speed_ms if speed_ms > 0 else 10.0
        
        # 3. Apply weather impact
        # Assuming 1.2 is the standard weather baseline; values above 1.2 increase travel time linearly.
        weather_multiplier = 1.0 + max(0.0, (weather_factor - 1.2) * 0.2)
        travel_time = base_time * weather_multiplier
        
        # 4. Core Energy Consumption Model
        # Part 1: Time-dependent baseline consumption (e.g., HVAC, electronics, idling).
        # Part 2: Speed-dependent power consumption (e.g., air resistance/drag).
        # Note: While congestion reduces speed, frequent stop-and-go behavior prevents energy cost 
        # from dropping linearly; we multiply by congestion_factor to reflect this efficiency loss.
        energy_cost = ((travel_time * 0.03) + (actual_speed_kmh ** 2) * 0.00005) * congestion_factor
        
        return energy_cost, travel_time

    @staticmethod
    def get_standard_time(distance_m, speed_kmh):
        """
        Provides a standard time reference for expert pathfinding/heuristics
        without considering stochastic weather or real-time congestion.
        """
        speed_ms = speed_kmh / 3.6
        return distance_m / speed_ms if speed_ms > 0 else 10.0