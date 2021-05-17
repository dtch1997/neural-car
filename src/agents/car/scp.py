import cvxpy as cp
import numpy as np

from src.utils import AttrDict
from src.envs.car import Environment as Env
from copy import deepcopy
from typing import *
from dataclasses import dataclass

def nxt(var: cp.Variable):
    return var[1:]
def curr(var: cp.Variable):
    return var[:-1]

@dataclass
class SCPAgent:
    """ Sequential convex programming controller, meant to be used with src.envs.car.Environment """
    # using @dataclass automates generating the __init__ function and setting these fields

    # Basic world parameters
    time_step_duration: float = 1 / Env.constants.fps # seconds. For 50 FPS
    num_time_steps_ahead: int = 200

    # The control problem is reparametrized with different control variables
    state_variable_names: Tuple[str] = Env.state_variable_names
    action_variable_names: Tuple[str] = Env.action_variable_names
    convergence_metrics: Tuple[str] = ("optimal_value")

    # Control parameters
    speed_limit = 10
    kappa_limit = np.tan(0.4) / Env.constants.ell
    accel_limit = 1
    pinch_limit = 3 / Env.constants.ell

    # Solver parameters
    solve_frequency: int = 20
    solve_tol: float = 1
    convergence_tol: float = 1e-2
    convergence_metric: str = 'optimal_value' 
    max_iters: int = 20
    solver = cp.ECOS

    # Obstacle parameters
    obstacle_radii: np.ndarray = None 
    obstacle_centers: np.ndarray = None
    using_obstacles: bool = False

    # Debug parameters 
    verbose: bool = False

    @staticmethod 
    def add_argparse_args(parser):
        parser.add_argument('--obstacle-radii', default = None)
        parser.add_argument('--obstacle-centers', default = None)
        parser.add_argument('--num-time-steps-ahead', type = int, default = 200)
        return parser 

    @staticmethod 
    def from_argparse_args(args):
        return SCPAgent(
            obstacle_radii = args.obstacle_radii,
            obstacle_centers = args.obstacle_centers,
            num_time_steps_ahead = args.num_time_steps_ahead
        )

    @property
    def num_states(self):
        return len(self.state_variable_names)
    @property
    def num_actions(self):
        return len(self.action_variable_names)

    def __post_init__(self):
        if self.convergence_metric not in self.convergence_metrics:
            raise ValueError(f"Convergence metric {self.convergence_metric} not recognized. Options: {self.convergence_metrics}")

    @property 
    def num_obstacles(self):
        return self._obstacle_centers.shape[0]

    def _setup_cp_problem(self):
        """ Set up the CVXPY problem once following DPP principles. 

        This must be called after setting or modifying any solver constants. 
        
        Solving just requires us to change parameter values. 
        In theory this allows CVXPY to solve the same problem multiple times at a greatly increased speed. 
        Reference: https://www.cvxpy.org/tutorial/advanced/index.html#disciplined-parametrized-programming
        """
        new_state_trajectory = cp.Variable((self.num_time_steps_ahead + 1, self.num_states))
        new_input_trajectory = cp.Variable((self.num_time_steps_ahead, self.num_actions))

        initial_state = cp.Parameter(shape = (self.num_states,))
        goal_state = cp.Parameter(shape = (self.num_states,))
        prev_state_trajectory = cp.Parameter((self.num_time_steps_ahead + 1, self.num_states))
        cos_prev_theta = cp.Parameter(self.num_time_steps_ahead + 1)
        sin_prev_theta = cp.Parameter(self.num_time_steps_ahead + 1) 
        obstacle_x_displacements = cp.Parameter((self.num_obstacles, self.num_time_steps_ahead+1))
        obstacle_y_displacements = cp.Parameter((self.num_obstacles, self.num_time_steps_ahead+1))
        obstacle_distances = cp.Parameter((self.num_obstacles, self.num_time_steps_ahead + 1))

        h = self.time_step_duration
        x = new_state_trajectory
        xt = prev_state_trajectory
        u = new_input_trajectory

        objective = cp.Minimize(
            self.time_step_duration * cp.sum_squares(u)
            + cp.norm(x[-1,:] - goal_state, p=1)
            + cp.sum(cp.maximum(x[:,3] - self.speed_limit, 0))
            + cp.sum(cp.maximum(x[:,4] - self.kappa_limit, 0))
            + cp.sum(cp.maximum(x[:,5] - self.accel_limit, 0))
            + cp.sum(cp.maximum(x[:,6] - self.pinch_limit, 0))
        )
        assert objective.is_dpp()

        constraints = []
        prev_theta = xt[:,2] # previous trajectory of angles
        prev_veloc = xt[:,3] # previous trajectory of velocities
        prev_kappa = xt[:,4] # previous trajectory of curvatures

        # Boundary conditions 
        constraints += [
            new_state_trajectory[0] == initial_state
        ]

        # Dynamics constraints 
        # This part is not DPP due to multiplication of parameters. 
        # We could make it DPP but it'd be quite a substantial effort and I'm not sure it's worthwhile
        # Also the obstacle avoidance constraints might end up not being DPP anyway
        constraints += [
            nxt(x[:,0]) == curr(x[:,0]) + h * ( #xpos constraint x[:,0]
                    cp.multiply(curr(x[:,3]), curr(cos_prev_theta))
                    - cp.multiply(cp.multiply(curr(prev_veloc), curr(sin_prev_theta)), curr(x[:,2] - prev_theta))
                ),
            nxt(x[:,1]) == curr(x[:,1]) + h * ( #ypos constraint x[:,1]
                    cp.multiply(curr(x[:,3]), curr(sin_prev_theta))
                    + cp.multiply(cp.multiply(curr(prev_veloc), curr(cos_prev_theta)), curr(x[:,2] - prev_theta))
                ),
            nxt(x[:,2]) == curr(x[:,2]) + h * ( #theta constraint x[:,2]
                    cp.multiply(curr(x[:,3]), curr(prev_kappa))
                    + cp.multiply(curr(prev_veloc), curr(x[:,4]) - curr(prev_kappa))
                ),
                
            nxt(x[:,3]) == curr(x[:,3]) + h * curr(x[:,5]), #velocity constraint x[:,3]
            nxt(x[:,4]) == curr(x[:,4]) + h * curr(x[:,6]), #kappa constraint x[:,4]
            nxt(x[:,5]) == curr(x[:,5]) + h * u[:,0], #acceleration constraint x[:,5]
            nxt(x[:,6]) == curr(x[:,6]) + h * u[:,1], #pinch constraint x[:,6]
        ]

        # Obstacle avoidance constraints
        rTotal = np.repeat(
            self._obstacle_radii, 
            repeats = self.num_time_steps_ahead + 1,
            axis = -1
        )
        xDiffs = x[:,0] - xt[:,0]
        yDiffs = x[:,1] - xt[:,1]
        xSub = obstacle_x_displacements
        ySub = obstacle_y_displacements
        zSubNorms = obstacle_distances

        
        for o in range(self.num_obstacles):
            constraints += [
                rTotal[o,:] \
                - zSubNorms[o,:] \
                - (cp.multiply(xSub[o,:], xDiffs) + cp.multiply(ySub[o,:], yDiffs)) / zSubNorms[o,:]
                <= 0
            ]
        

        # Set up cp.Problem
        problem = cp.Problem(objective, constraints)
        
        self.parameters = AttrDict.from_dict({
            'initial_state': initial_state,
            'goal_state': goal_state,
            'prev_state_trajectory': prev_state_trajectory, 
            'cos_prev_theta_trajectory': cos_prev_theta,
            'sin_prev_theta_trajectory': sin_prev_theta,
            'obstacle_distances': obstacle_distances,
            'obstacle_x_displacements': obstacle_x_displacements,
            'obstacle_y_displacements': obstacle_y_displacements,
        })
        self.variables = AttrDict.from_dict({
            'state_trajectory': new_state_trajectory,
            'input_trajectory': new_input_trajectory
        })
        self.problem = problem
        
    def reset(self, env):
        self._current_state = env.current_state
        self._goal_state = env.goal_state
        self._obstacle_centers = env.obstacle_centers
        self._obstacle_radii = env.obstacle_radii
        self._setup_cp_problem()
        self._input_trajectory = None 
        self._state_trajectory = None
        self._prev_input_trajectory = np.zeros((self.num_time_steps_ahead, self.num_actions)) 
        self._prev_state_trajectory = env.rollout_actions(self._current_state, self._prev_input_trajectory)
        self._steps_since_last_solve = 0

    def get_action(self, current_state, verbose = False):
        self._steps_since_last_solve += 1
        if self._input_trajectory is None \
            or np.linalg.norm(self._state_trajectory[self._steps_since_last_solve] - current_state) > self.solve_tol:
            # or self._steps_since_last_solve == self.solve_frequency:
            self._state_trajectory, self._input_trajectory = self._solve(
                current_state, 
                self._goal_state, 
                self._prev_state_trajectory, 
                self._prev_input_trajectory,
                verbose
            )
            self._steps_since_last_solve = 0        
        return self._input_trajectory[self._steps_since_last_solve]                


    def _solve(self, initial_state, goal_state, initial_state_trajectory, initial_input_trajectory, verbose = False):
        """ Perform one SCP solve to find an optimal trajectory """
        diff = self.convergence_tol + 1
        prev_state_trajectory = initial_state_trajectory
        prev_input_trajectory = initial_input_trajectory
        prev_optval = self.time_step_duration * np.linalg.norm(prev_input_trajectory,'fro')**2 + np.linalg.norm(goal_state- prev_state_trajectory[-1] ,1)

        for iteration in range(self.max_iters):
            # self._convex_solve is guaranteed to return a copy of state trajectory
            state_trajectory, input_trajectory, optval, status = self._convex_solve(initial_state, goal_state, prev_state_trajectory, prev_input_trajectory)
            if verbose: 
                print(f"SCP iteration {iteration}: status {status}, optval {optval}")
            # diff = np.abs(prev_optval - optval)
            diff = np.abs(prev_optval - optval).max()
            if diff < self.convergence_tol:
                break
            prev_state_trajectory = state_trajectory
            prev_input_trajectory = input_trajectory
            prev_optval = optval
        return state_trajectory, input_trajectory 

    def _convex_solve(self, initial_state, goal_state, prev_state_trajectory, prev_input_trajectory):
        self.parameters.initial_state.value = initial_state 
        self.parameters.goal_state.value = goal_state 
        self.parameters.prev_state_trajectory.value = prev_state_trajectory
        self.parameters.cos_prev_theta_trajectory.value = np.cos(prev_state_trajectory[:,2])
        self.parameters.sin_prev_theta_trajectory.value = np.sin(prev_state_trajectory[:,2])

        zObs = self._obstacle_centers 
        rObs = self._obstacle_radii
        zt = prev_state_trajectory[:,:2]
        zSub = np.transpose(np.repeat(np.expand_dims(zt,axis=2),self.num_obstacles,axis=2),axes=(2,0,1)) \
            - np.transpose(np.repeat(np.expand_dims(zObs,axis=2), self.num_time_steps_ahead + 1, axis=2),axes=(0,2,1)) #OxNx2 tensor containing differences between each position in the trajectory and obstacle center
        zSubNorms = np.linalg.norm(zSub,ord=2,axis=(-1)) # O x N matrix of Euclidean distances between each position in the trajectory and each obstacle center
        
        self.parameters.obstacle_x_displacements.value = zSub[:,:,0]
        self.parameters.obstacle_y_displacements.value = zSub[:,:,1]
        self.parameters.obstacle_distances.value = zSubNorms 

        optval = self.problem.solve(solver = self.solver)
        state_trajectory = self.variables.state_trajectory.value
        input_trajectory = self.variables.input_trajectory.value

        # Set to None to avoid accidentally reusing stale values in next solve 
        for pname, parameter in self.parameters.items():
            self.parameters[pname].value = None

        return state_trajectory, input_trajectory, optval, self.problem.status
