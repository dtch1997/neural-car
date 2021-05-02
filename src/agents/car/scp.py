import cvxpy as cp
import numpy as np

from envs.car import Environment as Env
from copy import deepcopy
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
    convergence_tol: float = 1e-7
    time_step_duration: float = 1 / Env.constants.fps # seconds. For 50 FPS
    num_time_steps_ahead: int = 200

    # The control problem is reparametrized with different control variables
    state_variable_names: Tuple[str] = Env.state_variable_names + ("kappa", "accel", "pinch")
    input_variable_names: Tuple[str] = ("jerk", "juke")

    # Control parameters
    speed_limit = 20
    kappa_limit = np.tan(0.4) / Env.constants.ell
    accel_limit = 1
    pinch_limit = 3 / Env.constants.ell

    # Obstacle parameters
    # TODO

    @property
    def num_states(self):
        return len(self.state_variable_names)
    @property
    def num_inputs(self):
        return len(self.input_variable_names)

    def __post_init__(self):
        self.state_trajectory = np.zeros((self.num_time_steps_ahead, self.num_states()))
        self.input_trajectory = np.zeros((self.num_time_steps_ahead, self.num_actions()))

    def init_obstacles(self, obstacle_centers, obstacle_radii):
        """ Initializes obstacles that the solver avoids 

        :param obstacle_centers: np.ndarray of shape (num_obstacles, 2)
        :param obstacle_radii: np.ndarray of shape (num_obstacles, 1)
        """
        self.obstacle_centers = obstacle_centers
        self.obstacle_radii = obstacle_radii

    def plan(self, state):
        pass

    def get_action(self, state):
        pass