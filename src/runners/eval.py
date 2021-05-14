import numpy as np

from pathlib import Path
from src.utils import plot_trajectory

OUTPUT_DIR = Path('output')

class EvaluationRunner:

    def __init__(self, args, env, agent):
        self.env = env
        self.agent = agent
        self.num_simulation_time_steps = args.num_simulation_time_steps
        self.save_filepath = args.save_filepath
        self.num_rollouts = args.num_rollouts
        self.dist_threshold = args.dist_threshold

    def get_savepath(self):
        return str(OUTPUT_DIR / self.save_filepath)

    @staticmethod 
    def add_argparse_args(parser):
        parser.add_argument("--num-simulation-time-steps", type=int, default=1200)
        parser.add_argument("--num-rollouts", type=int, default=5)
        parser.add_argument("--dist-threshold", type=int, default=1)
        parser.add_argument("--save-filepath", type=str, default = 'trajectory.npz')
        return parser 

    @staticmethod 
    def from_argparse_args(args, env, agent):
        return EvaluationRunner(args, env, agent)

    def run(self):
        actual_trajectory = np.zeros((self.num_simulation_time_steps + 1, self.agent.num_states))

        data =  np.zeros((self.num_rollouts * self.num_simulation_time_steps, self.agent.num_states+self.agent.num_actions))
        data_len = 0

        for i in range(self.num_rollouts):
            delta_x, delta_y, delta_th = (*10*np.random.rand(2),np.pi*np.random.rand()-np.pi/2)
            relative_goal = np.array([delta_x,delta_y,delta_th])
            self.env.reset(relative_goal) 
            self.agent.reset(self.env)

            initial_state = self.env.current_state
            current_state = self.env.current_state
            actual_trajectory[0] = current_state

            print(self.env.current_state)
            print(self.env.goal_state)
            for j in range(self.num_simulation_time_steps):
                self.env.render()
                action = self.agent.get_action(current_state)
                
                data[data_len] = np.concatenate([current_state, action])
                data_len += 1

                next_state, reward, done, info = self.env.take_action(action)            
                current_state = next_state
                actual_trajectory[j+1] = current_state

                diff = current_state[:3] - self.env.goal_state[:3]
                # TODO: Normalize theta to be between 0 and 2pi when calculating difference
                print(diff)
                if np.linalg.norm(diff).item() < self.dist_threshold:
                    break


            plot_trajectory(initial_state, self.env.goal_state, actual_trajectory[:j], str(OUTPUT_DIR / f'actual_trajectory_{i}.png'))

        OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
        np.save(self.get_savepath(), data)
        return actual_trajectory