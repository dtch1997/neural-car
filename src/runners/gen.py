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
                delta_x, delta_y, delta_th = (*np.random.uniform(low = -20, high = 20, size = 2), np.pi * np.random.uniform()-np.pi/2)
                relative_goal = np.array([delta_x,delta_y,delta_th])
                relative_obstacle_centers = np.random.uniform(low = -10, high = 10, size = (5, 2))
                obstacle_radii = np.ones(shape = (5, 1), dtype = np.float32)

                self.env.reset(relative_goal, relative_obstacle_centers, obstacle_radii, disable_view=True) 
                self.agent.reset(self.env)

                initial_state = self.env.current_state
                current_state = self.env.current_state

                grp = output_file.create_group(f'simulation_{i}')
                state_trajectory = grp.create_dataset(
                    f'state_trajectory', 
                    shape = (self.num_simulation_time_steps + 1, 7), 
                    dtype = 'f'
                )
                input_trajectory = grp.create_dataset(
                    f'input_trajectory',
                    shape = (self.num_simulation_time_steps, 2),
                    dtype = 'f'
                )
                grp.attrs['goal_state'] = self.env.goal_state
                grp.attrs['relative_obstacle_centers'] = relative_obstacle_centers
                grp.attrs['obstacle_centers'] = self.env.obstacle_centers
                grp.attrs['obstacle_radii'] = obstacle_radii

                state_trajectory[0] = initial_state

                self.log(f'Beginning simulation {i}')
                for j in range(self.num_simulation_time_steps):
                    try: 
                        action = self.agent.get_action(current_state)
                        next_state, reward, done, info = self.env.take_action(action)            
                        current_state = next_state

                        state_trajectory[j+1] = current_state
                        input_trajectory[j] = action

                        diff = self.env.goal_state[:3] - current_state[:3]
                        # Normalize theta to be between -pi and pi when calculating difference
                        while diff[2] > np.pi:
                            diff[2] -= 2 * np.pi
                        while diff[2] < -np.pi:
                            diff[2] += 2 * np.pi
                        
                        if np.linalg.norm(diff).item() < self.dist_threshold:
                            break
                    except Exception as e:
                        self.log(f"Rollout {i} exited with error {e}")
                        break          
                grp.attrs['num_steps'] = j
            return 
