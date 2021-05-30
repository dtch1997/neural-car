import h5py
from src.utils.viz import plot_trajectory

with h5py.File('datasets/simulation_output.hdf5','r') as file:
    print(file.keys())
    print(file['simulation_0'].keys())
    print(file['simulation_0'].attrs.keys())

    sim_0 = file['simulation_0']
    initial_state = sim_0['state_trajectory'][0]
    goal_state = sim_0.attrs['goal_state']
    state_trajectory = sim_0['state_trajectory']
    input_trajectory = sim_0['input_trajectory']
    obstacle_centers = sim_0.attrs['obstacle_centers']
    obstacle_radii = sim_0.attrs['obstacle_radii']
    num_steps = sim_0.attrs['num_steps']

    plot_trajectory(initial_state, goal_state, state_trajectory[:num_steps + 1], 'simulation_output_plot_0.png', 
            obstacle_centers, obstacle_radii)
    
    print(input_trajectory[:10])