import numpy as np
import pandas as pd

def build_uav_deeponet_engine(num_trajectories=1000, num_points_per_traj=100, output_filename='deeponet_dawama_data.csv'):
    """
    Generate custom dataset for DeepONet:
    - num_trajectories: Number of different vortex/wind scenarios (the input function 'u')
    - num_points_per_traj: Number of spatial evaluation points per scenario (the location 'y')
    Total data points = 1,000 * 100 = 100,000 samples
    """
    print(f"=== [START] UAV DeepONet Data Engine ===")
    
    # 1. Distribute the 8 sensors (fixed locations for the Branch Net)
    angles = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    sensor_positions = []
    for alpha in angles:
        sensor_positions.append([0.6 * np.cos(alpha), 0.6 * np.sin(alpha), 0.02]) # Outer sensors
        sensor_positions.append([0.3 * np.cos(alpha), 0.3 * np.sin(alpha), 0.01]) # Inner sensors
    sensor_positions = np.array(sensor_positions) # Shape: [8, 3]
    num_sensors = 8

    np.random.seed(57)
    rho_air = 1.225
    
    # Arrays to store the final DeepONet training triplets
    branch_inputs = [] # Input 'u' (readings from the 8 pressure sensors)
    trunk_inputs = []  # Input 'y' (the continuous spatial coordinates to predict velocity at)
    outputs = []       # Target 'G(u)(y)' (the resulting scalar velocity magnitude at location y)

    print("-> Generating trajectories and spatial evaluation points...")
    
    for t in range(num_trajectories):
        # Generate a random wind/vortex scenario (represents the continuous input function 'u')
        wind_vel = np.random.uniform(8.0, 35.0, (1, 3))
        vortex_cen = np.random.uniform(-1.2, 1.2, (1, 3))
        vortex_str = np.random.uniform(4.0, 25.0, (1, 1))
        vortex_rad = np.random.uniform(0.15, 0.6, (1, 1))
        
        # Calculate ground truth pressure at the 8 sensor locations for this specific scenario
        u_snapshot = np.zeros(num_sensors)
        for i in range(num_sensors):
            pos = sensor_positions[i, :]
            r_vec = vortex_cen - pos
            r_mag = np.linalg.norm(r_vec)
            
            induced_speed = (vortex_str * r_mag) / (2 * np.pi * (vortex_rad**2)) if r_mag < vortex_rad else vortex_str / (2 * np.pi * r_mag)
            total_v_sq = np.sum(wind_vel**2) + induced_speed**2
            u_snapshot[i] = 0.5 * rho_air * total_v_sq
            
        # Add sensor noise to simulate structural vibrations (Sim-to-Real gap)
        u_snapshot += np.random.normal(0, 12.5, num_sensors)
        
        # For each trajectory scenario (u), sample random spatial points (y) to evaluate the output
        for p in range(num_points_per_traj):
            # Random 3D coordinate around the UAV where wind velocity needs to be predicted
            y_point = np.random.uniform(-2.0, 2.0, 3) 
            
            # Calculate the actual wind speed at this query point (network's target output)
            r_vec_y = vortex_cen - y_point
            r_mag_y = np.linalg.norm(r_vec_y)
            induced_speed_y = (vortex_str * r_mag_y) / (2 * np.pi * (vortex_rad**2)) if r_mag_y < vortex_rad else vortex_str / (2 * np.pi * r_mag_y)
            
            # Compute the total scalar velocity magnitude at query point y
            total_velocity_at_y = np.sqrt(np.sum(wind_vel**2) + induced_speed_y**2)
            
            # Append the structured triplet data for DeepONet training
            branch_inputs.append(u_snapshot) # Stays constant for all 100 points within the same trajectory
            trunk_inputs.append(y_point)
            outputs.append(total_velocity_at_y)

    # 3. Compile everything into a structured Pandas DataFrame
    branch_inputs = np.array(branch_inputs)
    trunk_inputs = np.array(trunk_inputs)
    outputs = np.array(outputs)
    
    data_dict = {}
    for i in range(num_sensors):
        data_dict[f'Sensor_P{i+1}'] = branch_inputs[:, i]
        
    data_dict['Query_X'] = trunk_inputs[:, 0]
    data_dict['Query_Y'] = trunk_inputs[:, 1]
    data_dict['Query_Z'] = trunk_inputs[:, 2]
    data_dict['Target_Velocity'] = outputs.flatten()
    
    df = pd.DataFrame(data_dict)
    df.to_csv(output_filename, index=False)
    print(f"=== [SUCCESS] Generated DeepONet dataset shape: {df.shape} ===")
    return df

if __name__ == "__main__":
    df_deeponet = build_uav_deeponet_engine()