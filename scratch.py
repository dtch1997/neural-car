import h5py
import numpy as np
from src.utils.viz import plot_trajectory

with h5py.File('datasets/simulation_output.hdf5','r') as file:

    # TODO: Make this work
    for simulation_id in file.keys():
        partial_trajectory_list = []
        goal_states = []
        for goal_id in file[simulation_id].keys():
            print(f"{simulation_id}_{goal_id}")
            sim = file[simulation_id][goal_id]
            initial_state = sim['state_trajectory'][0]
            goal_state = sim.attrs['goal_state']
            goal_states.append(goal_state)
            partial_state_trajectory = sim['state_trajectory']
            partial_input_trajectory = sim['input_trajectory']
            obstacle_centers = sim.attrs['obstacle_centers']
            obstacle_radii = sim.attrs['obstacle_radii']
            num_steps = sim.attrs['num_steps']
            partial_trajectory_list.append(partial_state_trajectory[:num_steps + 1])


        state_trajectory = np.concatenate(partial_trajectory_list, axis=0)
        goal_states = np.vstack(goal_states)

        plot_trajectory(
            initial_state, 
            goal_states, 
            state_trajectory,
            partial_trajectory_list,
            f'datasets/{simulation_id}_{goal_id}.png', 
            obstacle_centers, obstacle_radii
        )