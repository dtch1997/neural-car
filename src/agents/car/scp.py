import cvxpy as cp
import numpy as np

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

    # Control parameters
    speed_limit = 20
    kappa_limit = np.tan(0.4) / Env.constants.ell
    accel_limit = 1
    pinch_limit = 3 / Env.constants.ell

    # Solver parameters
    convergence_tol: float = 1e-2
    max_iters: int = 100
    solver = cp.ECOS

    # Obstacle parameters
    obstacle_radii = None 
    obstacle_centers = None

    # Debug parameters 
    verbose: bool = False

    @property
    def num_states(self):
        return len(self.state_variable_names)
    @property
    def num_actions(self):
        return len(self.action_variable_names)

    def __post_init__(self):
        for var_idx, var_name in enumerate(self.state_variable_names):
            setattr(self, var_name+"idx", var_idx)

    def init_obstacles(self, obstacle_centers, obstacle_radii):
        """ Initializes obstacles that the solver avoids 

        :param obstacle_centers: np.ndarray of shape (num_obstacles, 2)
        :param obstacle_radii: np.ndarray of shape (num_obstacles, 1)
        """
        self.obstacle_centers = obstacle_centers
        self.obstacle_radii = obstacle_radii

    @property 
    def num_obstacles(self):
        if self.obstacle_centers is None:
            return 0
        return self.obstacles_centers.shape[0]

    def solve(self, initial_state, goal_state, initial_state_trajectory, initial_input_trajectory):
        """ Perform one SCP solve to find an optimal trajectory """
        diff = self.convergence_tol + 1
        prev_state_trajectory = initial_state_trajectory
        prev_input_trajectory = initial_input_trajectory
        prev_optval = self.time_step_duration * np.linalg.norm(prev_input_trajectory,'fro')**2 + np.linalg.norm(goal_state- prev_state_trajectory[-1] ,1)

        for iteration in range(self.max_iters):
            # self._convex_solve is guaranteed to return a copy of state trajectory
            state_trajectory, input_trajectory, optval, status = self._convex_solve(initial_state, goal_state, prev_state_trajectory)
            if self.verbose: 
                print(f"SCP iteration {iteration}: status {status}, optval {optval}")
            # diff = np.abs(prev_optval - optval)
            diff = np.abs(prev_optval - optval).max()
            if diff < self.convergence_tol:
                break
            prev_input_trajectory = input_trajectory
            prev_state_trajectory = state_trajectory
            prev_optval = optval
        return state_trajectory, input_trajectory 

    def _convex_solve(self, initial_state, goal_state, prev_state_trajectory):
        new_state_trajectory = cp.Variable((self.num_time_steps_ahead + 1, self.num_states))
        new_input_trajectory = cp.Variable((self.num_time_steps_ahead, self.num_actions))

        h = self.time_step_duration
        x = new_state_trajectory
        xt = prev_state_trajectory
        u = new_input_trajectory

        objective = cp.Minimize(
            self.time_step_duration * cp.square(cp.norm(u,'fro')) \
            + cp.norm(x[-1,:] - goal_state, p=1)
        )

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
                    cp.multiply(curr(x[:,3]), np.cos(curr(prev_theta)))
                    - cp.multiply(cp.multiply(curr(prev_veloc), np.sin(curr(prev_theta))), curr(x[:,2] - prev_theta))
                ),
            nxt(x[:,1]) == curr(x[:,1]) + h * ( #ypos constraint x[:,1]
                    cp.multiply(curr(x[:,3]), np.sin(curr(prev_theta)))
                    + cp.multiply(cp.multiply(curr(prev_veloc), np.cos(curr(prev_theta))), curr(x[:,2] - prev_theta))
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

        # Obstacle avoidance constraints
        """
        rObs = self.obstacle_radii 
        zObs = self.obstacle_centers
        zt = xt[:,:2]
        for o in range(0, self.num_obstacles): #constraints for each obstacle
            for i in range(1, self.num_time_steps_ahead): #apply constraints to mutable steps
                constraints += [rObs[o] - cp.norm((zt[i,:] - zObs[o,:])) - ((zt[i,:] - zObs[o,:]) / cp.norm((zt[i,:] - zObs[o,:]))) @ (x[i,:2]-zt[i,:]) <= 0]
        """

        problem = cp.Problem(objective, constraints)
        optval = problem.solve(solver = self.solver)
        return new_state_trajectory.value.copy(), new_input_trajectory.value.copy(), optval, problem.status