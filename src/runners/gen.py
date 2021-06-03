import numpy as np
import h5py

from pathlib import Path
from .eval import EvaluationRunner

class DataGenerationRunner(EvaluationRunner):

    @staticmethod 
    def from_argparse_args(args, env, agent):
        return DataGenerationRunner(args, env, agent)

    @property 
    def save_dir(self) -> Path:
        return Path('datasets')

    def run(self):
        np.random.seed(self.world_seed)
        save_path = str(self.save_dir / 'simulation_output.hdf5')
        self.save_dir.mkdir(exist_ok = True, parents = True)
        print(f"Saving to {save_path}")
        with h5py.File(save_path, 'w') as output_file:
            for i in range(self.num_rollouts):

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

                    self.env.update_obstacles(*self._generate_obstacles())
                    self.env.update_goal(self._generate_goal())
                    
                    self.agent.reset(self.env)

                    sub_grp.attrs['goal_state'] = self.env.goal_state
                    sub_grp.attrs['obstacle_centers'] = self.env.obstacle_centers
                    sub_grp.attrs['obstacle_radii'] = self.env.obstacle_radii

                    self.log(f'Beginning simulation {i} goal {j}')
                    for k in range(self.num_simulation_time_steps):
                        #self.env.render()
                        t += 1 # Increment the timer 
                        try: 
                            action, _ = self.agent.get_action(current_state)
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
