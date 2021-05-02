import numpy as np

from matplotlib import pyplot as plt
from src.utils import rotate_by_angle
from src.envs.car import Environment 
from src.agents.car import SCPAgent


def plot_trajectory(initial_state, goal_state, state_trajectory):
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
    plt.savefig('scp_trajectory.png')

def main():
    env = Environment()
    env.reset() 

    # Set up the initial state
    initial_state = env.current_state

    # Set up the final state
    x, y = initial_state[0], initial_state[1]
    theta = initial_state[2]
    direction = np.array([np.cos(theta), np.sin(theta),0,0])
    orth_direction = np.array([*rotate_by_angle(direction[:2], np.pi/2),0,0])
    goal_state = np.array([x, y, theta, 0, 0, 0, 0]) + 8 * np.hstack((direction,np.array([0,0,0]))) - 30 * np.hstack((orth_direction,np.array([0,0,0])))


    solver = SCPAgent(num_time_steps_ahead = 1200)
    # Set up a feasible initial trajectory
    zero_action = np.zeros((solver.num_time_steps_ahead, solver.num_actions)) 
    zero_action_state_trajectory = env.rollout_actions(initial_state, zero_action)
    state_trajectory, input_trajectory = solver.solve(initial_state, goal_state, zero_action_state_trajectory)
    plot_trajectory(initial_state, goal_state, state_trajectory)

if __name__ == "__main__":
    main()
    