import numpy as np

from pathlib import Path
from src.utils import plot_trajectory
from src.runners.train import MSERegression

OUTPUT_DIR = Path('output')

class EvaluationRunner:

    def __init__(self, args, env, agent):
        self.env = env
        self.agent = agent
        self.num_simulation_time_steps = args.num_simulation_time_steps
        self.save_filepath = args.save_filepath
        self.num_rollouts = args.num_rollouts
        self.dist_threshold = args.dist_threshold
        self.world_seed = args.world_seed

        if args.checkpoint_path is not None:
            self.agent = MSERegression.load_from_checkpoint(args.checkpoint_path, args = None, agent = agent)

    def get_savepath(self):
        return str(OUTPUT_DIR / self.save_filepath)

    @staticmethod 
    def add_argparse_args(parser):
        parser.add_argument("--num-simulation-time-steps", type=int, default=1200)
        parser.add_argument("--num-rollouts", type=int, default=5)
        parser.add_argument("--dist-threshold", type=int, default=1)
        parser.add_argument("--save-filepath", type=str, default = 'trajectory.npy')
        parser.add_argument("--checkpoint-path", type=str, default = None)
        parser.add_argument("--world-seed", type=int, default=None)
        return parser 

    def log(self, message: str):
        print(message)

    @staticmethod 
    def from_argparse_args(args, env, agent):
        return EvaluationRunner(args, env, agent)

    def run(self):
        OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
        actual_trajectory = np.zeros((self.num_simulation_time_steps + 1, self.env.num_states()))

        np.random.seed(self.world_seed)
        for i in range(self.num_rollouts):
            
            delta_x, delta_y, delta_th = (*np.random.uniform(low = -20, high = 20, size = 2), np.pi * np.random.uniform()-np.pi/2)
            relative_goal = np.array([delta_x,delta_y,delta_th])
            relative_obstacle_centers = np.random.uniform(low = -10, high = 10, size = (5, 2))
            obstacle_radii = np.ones(shape = (5, 1), dtype = np.float32)

            self.env.reset(relative_goal, relative_obstacle_centers, obstacle_radii) 
            self.agent.reset(self.env)

            initial_state = self.env.current_state
            current_state = self.env.current_state
            actual_trajectory[0] = current_state

            self.log(f'Beginning simulation {i}')
            for j in range(self.num_simulation_time_steps):
                self.env.render()
                action = self.agent.get_action(current_state)

                next_state, reward, done, info = self.env.take_action(action)            
                current_state = next_state
                actual_trajectory[j+1] = current_state

                diff = self.env.goal_state[:3] - current_state[:3]
                # Normalize theta to be between -pi and pi when calculating difference
                while diff[2] > np.pi:
                    diff[2] -= 2 * np.pi
                while diff[2] < -np.pi:
                    diff[2] += 2 * np.pi
                
                if np.linalg.norm(diff).item() < self.dist_threshold:
                    break

            plot_trajectory(
                initial_state = initial_state, 
                goal_state = self.env.goal_state, 
                state_trajectory = actual_trajectory[:j], 
                filepath = str(OUTPUT_DIR / f'actual_trajectory_{i}.png'), 
                obstacle_centers = self.env.obstacle_centers, 
                obstacle_radii = self.env.obstacle_radii
            )
        return actual_trajectory