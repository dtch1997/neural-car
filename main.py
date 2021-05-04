import numpy as np

from pathlib import Path
from matplotlib import pyplot as plt
from src.utils import rotate_by_angle
from src.envs.car import Environment 
from src.agents.car import SCPAgent

OUTPUT_DIR = Path('output')

def plot_trajectory(initial_state, goal_state, state_trajectory, filepath: str):
    fig, ax = plt.subplots(2,3)
    ax[0,0].scatter(state_trajectory[:,0], state_trajectory[:,1], c='black', label = 'Planned trajectory') #vehicle trajectory
    ax[0,0].scatter(goal_state[0], goal_state[1], s=30, c='red', label = 'Goal position')
    ax[0,0].scatter(initial_state[0], initial_state[1], s=30, c='blue', label='Initial position')
    ax[0,0].set_title("Position trajectory")

    time = np.arange(state_trajectory.shape[0])
    ax[0,1].plot(time, state_trajectory[:,3]) #velocity history
    ax[0,1].set_title("Velocity history")

    ax[0,2].plot(time, state_trajectory[:,5]) #acceleration history
    ax[0,2].set_title("Acceleration history")

    ax[1,0].plot(time, state_trajectory[:,2]) #vehicle attitude
    ax[1,0].set_title("Angle history")

    ax[1,1].plot(time, state_trajectory[:,4]) #curvature history
    ax[1,1].set_title("Curvature history")

    #ax[1,2].plot(time, np.arctan(ELL * xt[:,4])) #steering angle history
    #ax[1,2].set_title("Steering angle history")
    plt.savefig(filepath)

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
        num_time_steps_ahead = 600,    
        convergence_tol = 1e-2,
        max_iters = 3, 
        verbose = False
    )

    # Set up a feasible initial trajectory
    zero_action = np.zeros((solver.num_time_steps_ahead, solver.num_actions)) 
    zero_action_state_trajectory = env.rollout_actions(initial_state, zero_action)
    
    # Begin the simulation
    num_simulation_time_steps = 300
    actual_trajectory = np.zeros((num_simulation_time_steps, 7))
    prev_state_trajectory = zero_action_state_trajectory
    prev_input_trajectory = zero_action  
    current_state = initial_state  
    for i in range(num_simulation_time_steps):
        env.render()
        state_trajectory, input_trajectory = solver.solve(current_state, goal_state, prev_state_trajectory, prev_input_trajectory)
        # Plot the first iteration of planned trajectories
        if i == 0:
            plot_trajectory(initial_state, goal_state, state_trajectory, filepath = str(OUTPUT_DIR/'scp_trajectory.png'))
        prev_state_trajectory = np.concatenate([state_trajectory[1:], state_trajectory[-1][np.newaxis, :]], axis=0) 
        prev_input_trajectory = np.concatenate([input_trajectory[1:], input_trajectory[-1][np.newaxis, :]], axis=0) 
        next_state = env.take_action(input_trajectory[0])
        actual_trajectory[i] = next_state
        current_state = next_state.copy()

    plot_trajectory(initial_state, goal_state, actual_trajectory, filepath = str(OUTPUT_DIR / 'actual_trajectory.png'))

if __name__ == "__main__":
    main()
    