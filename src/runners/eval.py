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
        self.rearward_goals = args.rearward_goals
        self.num_goals = args.num_goals

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
        parser.add_argument("--rearward-goals", type=bool, default=False)
        parser.add_argument("--num-goals", type=int, default=1)
        return parser 

    def log(self, message: str):
        print(message)

    @staticmethod 
    def from_argparse_args(args, env, agent):
        return EvaluationRunner(args, env, agent)

    def run(self):
        OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
        

        np.random.seed(self.world_seed)
        for i in range(self.num_rollouts):
            #delta_x, delta_y, delta_th = (*np.random.uniform(low = -20, high = 20, size = 2), np.pi * np.random.uniform()-np.pi/2)
            #relative_goal = np.array([delta_x,delta_y,delta_th])
            #relative_obstacle_centers = np.random.uniform(low = -10, high = 10, size = (5, 2))
            #obstacle_radii = np.ones(shape = (5, 1), dtype = np.float32)

            actual_trajectory = np.zeros(((self.num_simulation_time_steps + 1)*self.num_goals, self.env.num_states()))

            # Reset environment. Done once per rollout
            self.env.reset(disable_view=False) 

            current_state = self.env.current_state
            initial_state = self.env.current_state
            actual_trajectory[0] = current_state

            t = -1 # Keeps track of how many steps have passed

            goal_states = np.array([])
            obstacle_centers = np.array([])
            obstacle_radiis = np.array([])

            partial_trajectory_list = []

            for j in range(self.num_goals):
                partial_trajectory = np.zeros(((self.num_simulation_time_steps + 1), self.env.num_states()))
                
                if not self.rearward_goals:
                    r, phi, delta_th = (30 * np.random.uniform(), np.pi * np.random.uniform()-np.pi/2, np.pi * np.random.uniform()-np.pi/2)
                else:
                    r, phi, delta_th = (30 * np.random.uniform(), 2*np.pi * np.random.uniform()-np.pi, np.pi * np.random.uniform()-np.pi/2)

                relative_goal = np.array([r, phi, delta_th])
                relative_obstacle_centers = np.random.uniform(low = -10, high = 10, size = (5, 2))
                obstacle_radii = np.ones(shape = (5, 1), dtype = np.float32)
                
                self.env.update_goal(relative_goal)
                self.env.update_obstacles(relative_obstacle_centers, obstacle_radii)
                
                goal_states = np.vstack([goal_states, self.env.goal_state]) if goal_states.size else self.env.goal_state
                obstacle_centers = np.vstack([obstacle_centers, self.env.obstacle_centers]) if obstacle_centers.size else self.env.obstacle_centers
                obstacle_radiis = np.vstack([obstacle_radiis, self.env.obstacle_radii]) if obstacle_radiis.size else self.env.obstacle_radii
                
                self.agent.reset(self.env)

                self.log(f'Beginning simulation {i} goal {j}')
                for k in range(self.num_simulation_time_steps):
                    self.env.render()
                    t += 1 # Increment the timer
                    try: 
                        action = self.agent.get_action(current_state)
                        next_state, reward, done, info = self.env.take_action(action)            
                        current_state = next_state
                        actual_trajectory[t+1] = current_state
                        partial_trajectory[k+1] = current_state

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

                partial_trajectory_list.append(partial_trajectory[:k+1])
            plot_trajectory(
                initial_state = initial_state, 
                goal_states = goal_states, 
                state_trajectory = actual_trajectory[:t+1], 
                partial_trajectory_list = partial_trajectory_list,
                filepath = str(OUTPUT_DIR / f'actual_trajectory_{i}.png'), 
                obstacle_centers = obstacle_centers, 
                obstacle_radii = obstacle_radiis
            )
        return actual_trajectory
        