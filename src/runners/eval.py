import numpy as np

from pathlib import Path
from src.utils import plot_trajectory

OUTPUT_DIR = Path('output')

class EvaluationRunner:

    def __init__(self, args, env, agent):
        self.env = env
        self.agent = agent
        self.num_simulation_time_steps = 1200

    @staticmethod 
    def add_argparse_args(parser):
        return parser 

    @staticmethod 
    def from_argparse_args(args, env, agent):
        return EvaluationRunner(args, env, agent)

    def run(self):
        self.env.reset() 
        self.agent.reset(self.env)

        actual_trajectory = np.zeros((self.num_simulation_time_steps + 1, 7))
        initial_state = self.env.current_state
        current_state = self.env.current_state
        actual_trajectory[0] = current_state

        for i in range(self.num_simulation_time_steps):
            self.env.render()
            action = self.agent.get_action(current_state)
            next_state, reward, done, info = self.env.take_action(action)            
            current_state = next_state
            actual_trajectory[i+1] = current_state

        OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
        plot_trajectory(initial_state, self.env.goal_state, actual_trajectory, str(OUTPUT_DIR / 'actual_trajectory.png'))
        return actual_trajectory