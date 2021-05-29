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
                # By default the state/input trajectories assume max possible simulation steps 
                state_trajectory = grp.create_dataset(
                    f'state_trajectory', 
                    shape = ((self.num_simulation_time_steps + 1)*self.num_goals, 7), 
                    dtype = 'f'
                )
                input_trajectory = grp.create_dataset(
                    f'input_trajectory',
                    shape = ((self.num_simulation_time_steps)*self.num_goals, 2),
                    dtype = 'f'
                )
                grp.attrs['goal_state'] = np.array([])
                grp.attrs['relative_obstacle_centers'] = np.array([])
                grp.attrs['obstacle_centers'] = np.array([])
                grp.attrs['obstacle_radii'] = np.array([])

                current_state = self.env.current_state
                initial_state = self.env.current_state
                state_trajectory[0] = initial_state

                t = -1 # Keeps track of how many steps have passed

                for j in range(self.num_goals):
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

                    grp.attrs['goal_state'] = np.vstack([grp.attrs['goal_state'], self.env.goal_state]) if grp.attrs['goal_state'].size else self.env.goal_state
                    grp.attrs['relative_obstacle_centers'] = np.vstack([grp.attrs['relative_obstacle_centers'], relative_obstacle_centers]) if grp.attrs['relative_obstacle_centers'].size else relative_obstacle_centers
                    grp.attrs['obstacle_centers'] = np.vstack([grp.attrs['obstacle_centers'], self.env.obstacle_centers]) if grp.attrs['obstacle_centers'].size else self.env.obstacle_centers
                    grp.attrs['obstacle_radii'] = np.vstack([grp.attrs['obstacle_radii'], obstacle_radii]) if grp.attrs['obstacle_radii'].size else obstacle_radii

                    self.log(f'Beginning simulation {i} goal {j}')
                    for k in range(self.num_simulation_time_steps):
                        t += 1 # Increment the timer
                        try: 
                            action = self.agent.get_action(current_state)
                            next_state, reward, done, info = self.env.take_action(action)            
                            current_state = next_state
    
                            state_trajectory[t+1] = current_state
                            input_trajectory[t] = action
    
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
                    grp.attrs['num_steps'] = j
            return 
