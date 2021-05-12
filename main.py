import numpy as np

from pathlib import Path
from matplotlib import pyplot as plt

from src.utils import rotate_by_angle, plot_trajectory
from src.envs.car import Environment 
from src.agents.car import SCPAgent

OUTPUT_DIR = Path('output')

def main():
    # Create output directory if it doesn't exist yet
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents = True, exist_ok = True)

    env = Environment(    
        allow_reverse=True,
        grayscale=1,
        show_info_panel=1,
        discretize_actions=None,
        num_obstacles=100,
        num_tracks=1,
        num_lanes=1,
        num_lanes_changes=4,
        max_time_out=0,
        frames_per_state=4
    )
    env.reset() 

    # Set up the initial state
    initial_state = env.current_state

    # Set up the final state
    x, y = initial_state[0], initial_state[1]
    theta = initial_state[2]
    direction = np.array([np.cos(theta), np.sin(theta),0,0])
    orth_direction = np.array([*rotate_by_angle(direction[:2], np.pi/2),0,0])
    goal_state = np.array([x, y, theta, 0, 0, 0, 0]) + 8 * np.hstack((direction,np.array([0,0,0]))) - 30 * np.hstack((orth_direction,np.array([0,0,0])))

    solver = SCPAgent(
        num_time_steps_ahead = 300,    
        convergence_tol = 1e-2,
        convergence_metric = "optimal_value",
        max_iters = 3, 
        verbose = True
    )

    # Set up a feasible initial trajectory
    zero_action = np.zeros((solver.num_time_steps_ahead, solver.num_actions)) 
    zero_action_state_trajectory = env.rollout_actions(initial_state, zero_action)
    
    # Begin the simulation
    solve_frequency = 20
    num_simulation_time_steps = 1200
    actual_trajectory = np.zeros((num_simulation_time_steps, 7))
    prev_state_trajectory = zero_action_state_trajectory
    prev_input_trajectory = zero_action  
    current_state = initial_state  
    for i in range(num_simulation_time_steps):
        env.render()
        if i % solve_frequency == 0:
            state_trajectory, input_trajectory = solver.solve(current_state, goal_state, prev_state_trajectory, prev_input_trajectory)
        # Plot the first iteration of planned trajectories
        if i == 0:
            plot_trajectory(initial_state, goal_state, state_trajectory, filepath = str(OUTPUT_DIR/ 'optimized_scp_trajectory.png'))
        prev_state_trajectory = np.concatenate([state_trajectory[1:], state_trajectory[-1][np.newaxis, :]], axis=0) 
        prev_input_trajectory = np.concatenate([input_trajectory[1:], input_trajectory[-1][np.newaxis, :]], axis=0) 
        next_state, reward, done, info = env.take_action(input_trajectory[i % solve_frequency])
        actual_trajectory[i] = next_state
        current_state = next_state.copy()

    plot_trajectory(initial_state, goal_state, actual_trajectory, filepath = str(OUTPUT_DIR / 'optimized_actual_trajectory.png'))

if __name__ == "__main__":
    main()
    