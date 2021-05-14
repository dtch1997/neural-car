import numpy as np

from pathlib import Path
from src.utils import plot_trajectory

OUTPUT_DIR = Path('output')

class EvaluationRunner:

    def __init__(self, args, env, agent):
        self.env = env
        self.agent = agent
        self.num_simulation_time_steps = args.num_simulation_time_steps
        self.saved_filepath = args.save_filepath
        self.num_rollouts = args.num_rollouts
        self.dist_threshold = args.dist_threshold

    @staticmethod 
    def add_argparse_args(parser):
        parser.add_argument("--num-simulation-time-steps", type=int, default=1200)
        parser.add_argument("--save-filepath", default="saved_data.csv")
        parser.add_argument("--num-rollouts", type=int, default=5)
        parser.add_argument("--dist-threshold", type=int, default=0.2)
        return parser 

    @staticmethod 
    def from_argparse_args(args, env, agent):
        return EvaluationRunner(args, env, agent)

    def run(self):
        delta_x, delta_y, delta_th = (10,10,np.pi/4)
        relative_goal = np.array([delta_x,delta_y,delta_th])
        self.env.reset(relative_goal) 
        self.agent.reset(self.env)

        actual_trajectory = np.zeros((self.num_simulation_time_steps + 1, self.agent.num_states))
        initial_state = self.env.current_state
        current_state = self.env.current_state
        actual_trajectory[0] = current_state

        data =  np.zeros((1, self.agent.num_states+self.agent.num_actions))

        for i in range(self.num_rollouts):
            for i in range(self.num_simulation_time_steps):
                self.env.render()
                action = self.agent.get_action(current_state)
                if i == 0:
                    data = np.hstack(current_state, action)
                else:
                    data = np.vstack(data, np.hstack(current_state, action))
                next_state, reward, done, info = self.env.take_action(action)            
                current_state = next_state
                actual_trajectory[i+1] = current_state
                if np.linalg.norm(current_state[:3], relative_goal) < self.dist_threshold:
                    break
            delta_x, delta_y, delta_th = (*10*np.random.rand(2),np.pi*np.random.rand()-np.pi/2)
            relative_goal = np.array([delta_x,delta_y,delta_th])
            self.env.reset(relative_goal)

        np.savetxt("saved_data.csv", data, delimiter=",")
        OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
        plot_trajectory(initial_state, self.env.goal_state, actual_trajectory, str(OUTPUT_DIR / 'actual_trajectory.png'))
        return actual_trajectory