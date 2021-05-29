import matplotlib.pyplot as plt
import numpy as np

def plot_trajectory(initial_state, goal_states, state_trajectory, partial_trajectory_list,
                    filepath: str, obstacle_centers, obstacle_radii, plot_obstacles=True):
    
    goal_states = np.atleast_2d(goal_states)
    fig, ax = plt.subplots(2,3)
    ax[0,0].scatter(state_trajectory[:,0], state_trajectory[:,1], s=5, c='black', label = 'Planned trajectory') #vehicle trajectory
    
    ax[0,0].scatter(initial_state[0], initial_state[1], s=30, c='blue', label='Initial position')
    
    if plot_obstacles:
        for o in range(obstacle_centers.shape[0]):
            circle = plt.Circle((obstacle_centers[o,:]), obstacle_radii[o], color = 'green')
            ax[0,0].add_patch(circle)
    
    #ax[0,0].scatter(obstacle_centers[:,0], obstacle_centers[:,1], s = obstacle_radii.reshape(-1) * 10, c='green', label = 'Obstacles')
    ax[0,0].set_title("Position trajectory")
    
    time = np.arange(state_trajectory.shape[0])
    
    ax[0,0].scatter(goal_states[:,0], goal_states[:,1], s=30, c='red', label = 'Goal positions')
    
    ax[0,1].plot(time, state_trajectory[:,3], label = 'Trajectory') #velocity history
    vel_goals = [np.ones(traj.shape[0]) * goal_states[i,3] for i, traj in enumerate(partial_trajectory_list)]
    
    ax[0,1].plot(time, np.hstack(vel_goals), label= 'Goal')
    ax[0,1].legend()
    ax[0,1].set_title("Velocity history")
    
    ax[0,2].plot(time, state_trajectory[:,5], label = 'Trajectory') #acceleration history
    acc_goals = [np.ones(traj.shape[0]) * goal_states[i,5] for i, traj in enumerate(partial_trajectory_list)]
    ax[0,2].plot(time, np.hstack(acc_goals), label= 'Goal')
    ax[0,2].legend()    
    ax[0,2].set_title("Acceleration history")
    
    ax[1,0].plot(time, state_trajectory[:,2], label = 'Trajectory') #vehicle attitude
    attitude_goals = [np.ones(traj.shape[0]) * goal_states[i,2] for i, traj in enumerate(partial_trajectory_list)]
    ax[1,0].plot(time, np.hstack(attitude_goals), label= 'Goal')
    ax[1,0].legend()
    ax[1,0].set_title("Angle history")
    
    ax[1,1].plot(time, state_trajectory[:,4], label = 'Trajectory') #curvature history
    curvature_goals = [np.ones(traj.shape[0]) * goal_states[i,4] for i, traj in enumerate(partial_trajectory_list)]
    ax[1,1].plot(time, np.hstack(curvature_goals), label= 'Goal')
    ax[1,1].legend()
    ax[1,1].set_title("Curvature history")
    
    plt.savefig(filepath)
    
