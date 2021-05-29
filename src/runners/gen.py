import numpy as np
import h5py
from .eval import EvaluationRunner

class DataGenerationRunner(EvaluationRunner):


    @staticmethod 
    def from_argparse_args(args, env, agent):
        return DataGenerationRunner(args, env, agent)

    def run(self):
        np.random.seed(self.world_seed)
        with h5py.File('simulation_output.hdf5', 'w') as output_file:
            for i in range(self.num_rollouts):
                #delta_x, delta_y, delta_th = (*np.random.uniform(low = -20, high = 20, size = 2), np.pi * np.random.uniform()-np.pi/2)
                #relative_goal = np.array([delta_x,delta_y,delta_th])
                #relative_obstacle_centers = np.random.uniform(low = -10, high = 10, size = (5, 2))
                #obstacle_radii = np.ones(shape = (5, 1), dtype = np.float32)

                # Reset environment. Done once per rollout
                self.env.reset(disable_view=True) 

                grp = output_file.create_group(f'simulation_{i}')
                grp.attrs['end_markers'] = np.array([])
                grp.attrs['num_total_steps'] = 0
                
                t = -1 # Keeps track of how many steps have passed

                for j in range(self.num_goals):
                    
                    sub_grp = grp.create_group(f'goal_{j}')
                    
                    # By default the state/input trajectories assume max possible simulation steps 
                    state_trajectory = sub_grp.create_dataset(
                        f'state_trajectory', 
                        shape = (self.num_simulation_time_steps + 1, 7), 
                        dtype = 'f'
                    )
                    input_trajectory = sub_grp.create_dataset(
                        f'input_trajectory',
                        shape = (self.num_simulation_time_steps, 2),
                        dtype = 'f'
                    )
                    
                    current_state = self.env.current_state 
                    state_trajectory[0] = current_state

                    if not self.rearward_goals:
                        r, phi, delta_th = (30 * np.random.uniform(), np.pi * np.random.uniform()-np.pi/2, np.pi * np.random.uniform()-np.pi/2)
                    else:
                        r, phi, delta_th = (30 * np.random.uniform(), 2*np.pi * np.random.uniform()-np.pi, np.pi * np.random.uniform()-np.pi/2)
    
                    relative_goal = np.array([r, phi, delta_th])
                    relative_obstacle_centers = np.random.uniform(low = -10, high = 10, size = (5, 2))
                    obstacle_radii = np.ones(shape = (5, 1), dtype = np.float32)
                    
                    self.env.update_goal(relative_goal)
                    self.env.update_obstacles(relative_obstacle_centers, obstacle_radii)
                    self.agent.reset(self.env)

                    sub_grp.attrs['goal_state'] = self.env.goal_state
                    sub_grp.attrs['relative_obstacle_centers'] = relative_obstacle_centers
                    sub_grp.attrs['obstacle_centers'] = self.env.obstacle_centers
                    sub_grp.attrs['obstacle_radii'] = obstacle_radii

                    self.log(f'Beginning simulation {i} goal {j}')
                    for k in range(self.num_simulation_time_steps):
                        t += 1 # Increment the timer 
                        try: 
                            action = self.agent.get_action(current_state)
                            next_state, reward, done, info = self.env.take_action(action)            
                            current_state = next_state
    
                            state_trajectory[k+1] = current_state
                            input_trajectory[k] = action
    
                            diff = self.env.goal_state[:3] - current_state[:3]
                            # Normalize theta to be between -pi and pi when calculating difference
                            while diff[2] > np.pi:
                                diff[2] -= 2 * np.pi
                            while diff[2] < -np.pi:
                                diff[2] += 2 * np.pi
                            
                            if np.linalg.norm(diff).item() < self.dist_threshold:
                                break
                        except Exception as e:
                            self.log(f"Rollout {i} on goal {j} exited with error {e}")
                            break          
                    
                    sub_grp.attrs['num_steps'] = k
                    grp.attrs['num_total_steps'] += k
                    grp.attrs['end_markers'] = np.hstack([grp.attrs['end_markers'], t]) if grp.attrs['end_markers'].size else np.array([t])
            return 
