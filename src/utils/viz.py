import matplotlib.pyplot as plt
import numpy as np

def plot_trajectory(initial_state, goal_state, state_trajectory, filepath: str):
    fig, ax = plt.subplots(2,3)
    ax[0,0].scatter(state_trajectory[:,0], state_trajectory[:,1], c='black', label = 'Planned trajectory') #vehicle trajectory
    ax[0,0].scatter(goal_state[0], goal_state[1], s=30, c='red', label = 'Goal position')
    ax[0,0].scatter(initial_state[0], initial_state[1], s=30, c='blue', label='Initial position')
    ax[0,0].set_title("Position trajectory")

    time = np.arange(state_trajectory.shape[0])
    ax[0,1].plot(time, state_trajectory[:,3], label = 'Trajectory') #velocity history
    ax[0,1].plot(time, np.ones_like(time) * goal_state[3], label= 'Goal')
    ax[0,1].legend()
    ax[0,1].set_title("Velocity history")

    ax[0,2].plot(time, state_trajectory[:,5], label = 'Trajectory') #acceleration history
    ax[0,2].plot(time, np.ones_like(time) * goal_state[5], label= 'Goal')
    ax[0,2].legend()    
    ax[0,2].set_title("Acceleration history")

    ax[1,0].plot(time, state_trajectory[:,2], label = 'Trajectory') #vehicle attitude
    ax[1,0].plot(time, np.ones_like(time) * goal_state[2], label= 'Goal')
    ax[1,0].legend()
    ax[1,0].set_title("Angle history")

    ax[1,1].plot(time, state_trajectory[:,4], label = 'Trajectory') #curvature history
    ax[1,1].plot(time, np.ones_like(time) * goal_state[4], label= 'Goal')
    ax[1,1].legend()
    ax[1,1].set_title("Curvature history")

    plt.savefig(filepath)