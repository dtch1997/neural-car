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
    speed_limit = 20
    kappa_limit = np.tan(0.4) / Env.constants.ell
    accel_limit = 1
    pinch_limit = 3 / Env.constants.ell

    # Solver parameters
    solve_frequency: int = 20
    convergence_tol: float = 1e-2
    convergence_metric: str = 'optimal_value' 
    max_iters: int = 3
    solver = cp.ECOS

    # Obstacle parameters
    obstacle_radii = None 
    obstacle_centers = None
    using_obstacles = False

    # Debug parameters 
    verbose: bool = False

    @staticmethod 
    def add_argparse_args(parser):
        return parser 

    @staticmethod 
    def from_argparse_args(parser):
        return SCPAgent()

    @property
    def num_states(self):
        return len(self.state_variable_names)
    @property
    def num_actions(self):
        return len(self.action_variable_names)

    def __post_init__(self):
        if self.convergence_metric not in self.convergence_metrics:
            raise ValueError(f"Convergence metric {self.convergence_metric} not recognized. Options: {self.convergence_metrics}")
        self._setup_cp_problem()

    def init_obstacles(self, obstacle_centers, obstacle_radii):
        """ Initializes obstacles that the solver avoids 

        :param obstacle_centers: np.ndarray of shape (num_obstacles, 2)
        :param obstacle_radii: np.ndarray of shape (num_obstacles, 1)
        """
        self.obstacle_centers = obstacle_centers
        self.obstacle_radii = obstacle_radii
        self._setup_cp_problem()

    @property 
    def num_obstacles(self):
        if self.obstacle_centers is None:
            return 0
        return self.obstacle_centers.shape[0]

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

        h = self.time_step_duration
        x = new_state_trajectory
        xt = prev_state_trajectory
        u = new_input_trajectory

        objective = cp.Minimize(
            self.time_step_duration * cp.sum_squares(u) \
            + cp.norm(x[-1,:] - goal_state, p=1)
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

        # Control limit constraints
        constraints += [
            cp.norm(x[:,3], p=np.inf)  <= self.speed_limit, #max forwards velocity (speed limit)
            cp.norm(x[:,4], p=np.inf)  <= self.kappa_limit, #maximum curvature
            cp.norm(x[:,5], p=np.inf) <= self.accel_limit, #max acceleration
            cp.norm(x[:,6], p=np.inf) <= self.pinch_limit #max pinch
        ]

        # TODO: Obstacle constraints
        if self.obstacle_radii is not None and self.obstacle_centers is not None:
            rObs = self.obstacle_radii 
            zObs = self.obstacle_centers
            zt = xt[:,:2]
            for o in range(0, self.num_obstacles): #constraints for each obstacle
                for i in range(1, self.num_time_steps_ahead): #apply constraints to mutable steps
                    constraints += [rObs[o] - cp.norm((zt[i,:] - zObs[o,:])) - ((zt[i,:] - zObs[o,:]) / cp.norm((zt[i,:] - zObs[o,:]))) @ (x[i,:2]-zt[i,:]) <= 0]

        problem = cp.Problem(objective, constraints)
        self.parameters = AttrDict.from_dict({
            'initial_state': initial_state,
            'goal_state': goal_state,
            'prev_state_trajectory': prev_state_trajectory, 
            'cos_prev_theta_trajectory': cos_prev_theta,
            'sin_prev_theta_trajectory': sin_prev_theta
        })
        self.variables = AttrDict.from_dict({
            'state_trajectory': new_state_trajectory,
            'input_trajectory': new_input_trajectory
        })
        self.problem = problem

    def reset(self, env):
        self._current_state = env.current_state
        self._goal_state = env.goal_state
        self._prev_input_trajectory = np.zeros((self.num_time_steps_ahead, self.num_actions)) 
        self._prev_state_trajectory = env.rollout_actions(self._current_state, self._prev_input_trajectory)
        self._time = 0

    def get_action(self, current_state):
        def forward(trajectory: np.ndarray, num_time_steps: int):
            """ Advance num_time_steps along a trajectory, copying the last value """
            return np.concatenate([trajectory[num_time_steps:]] + [trajectory[-1][np.newaxis,:]] * num_time_steps)
        if self._time % self.solve_frequency == 0:
            self._state_trajectory, self._input_trajectory = self._solve(
                current_state, 
                self._goal_state, 
                self._prev_state_trajectory, 
                self._prev_input_trajectory
            )
        self._time += 1
        return self._input_trajectory[self._time % self.solve_frequency]

    def _solve(self, initial_state, goal_state, initial_state_trajectory, initial_input_trajectory):
        """ Perform one SCP solve to find an optimal trajectory """
        diff = self.convergence_tol + 1
        prev_state_trajectory = initial_state_trajectory
        prev_input_trajectory = initial_input_trajectory
        prev_optval = self.time_step_duration * np.linalg.norm(prev_input_trajectory,'fro')**2 + np.linalg.norm(goal_state- prev_state_trajectory[-1] ,1)

        for iteration in range(self.max_iters):
            # self._convex_solve is guaranteed to return a copy of state trajectory
            state_trajectory, input_trajectory, optval, status = self._convex_solve(initial_state, goal_state, prev_state_trajectory, prev_input_trajectory)
            if self.verbose: 
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

        optval = self.problem.solve(solver = self.solver)
        state_trajectory = self.variables.state_trajectory.value
        input_trajectory = self.variables.input_trajectory.value

        # Set to None to avoid accidentally reusing stale values in next solve 
        for pname, parameter in self.parameters.items():
            self.parameters[pname].value = None

        return state_trajectory, input_trajectory, optval, self.problem.status
