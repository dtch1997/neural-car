import numpy as np

from .car_racing import CarRacing, FPS
from .dynamics import SIZE
from src.utils import AttrDict, rotate_by_angle
from typing import *


class Environment(CarRacing):
    """ A wrapper around the CarRacing environment 
    
    Provides APIs for: 
    1. Extracting state variables from openAI gym
    2. Converting inputs back to openAI gym controls
    """ 
    constants = AttrDict.from_dict({
        "size": SIZE, 
        "ell": SIZE * (80 + 82),
        "fps": FPS # Number of frames per second in the simulation
    })

    state_variable_names = ("xpos", "ypos", "theta", "velocity", "kappa", "accel", "pinch")
    action_variable_names = ("jerk", "juke")

    def __init__(self):
        # Fix these options for now. We can support alternate options in the future
        super(Environment, self).__init__(            
            allow_reverse=True,
            grayscale=1,
            show_info_panel=1,
            discretize_actions=None,
            num_obstacles=100,
            num_tracks=1,
            num_lanes=1,
            num_lanes_changes=4,
            max_time_out=0,
            frames_per_state=4
        )

    @staticmethod 
    def add_argparse_args(parser):
        return parser 
    
    @staticmethod 
    def from_argparse_args(args):
        return Environment()

    @property 
    def time_step_duration(self):
        return 1 / self.constants.fps

    @staticmethod
    def num_states():
        return len(Environment.state_variable_names)
    
    @staticmethod 
    def num_actions():
        return len(Environment.action_variable_names)

    @property 
    def current_state(self):
        return self._current_state.copy()

    @property 
    def goal_state(self):
        return self._goal_state.copy()

    @property 
    def obstacle_centers(self):
        return self._obstacle_centers.copy()

    @property 
    def obstacle_radii(self):
        return self._obstacle_radii.copy()

    def disable_view_window(self):
        from gym.envs.classic_control import rendering
        org_constructor = rendering.Viewer.__init__
    
        def constructor(self, *args, **kwargs):
            org_constructor(self, *args, **kwargs)
            self.window.set_visible(visible=False)
    
        rendering.Viewer.__init__ = constructor

    def reset(self, relative_goal, obstacle_centers = None, obstacle_radii = None, disable_view=False):
        if disable_view:
            self.disable_view_window() 
        super(Environment, self).reset()

        # When resetting, the acceleration and pinch are always zero
        self._current_state = np.concatenate([self._get_env_vars(), np.zeros(2)])
        self._obstacle_centers = obstacle_centers + self._current_state[:2]
        self._obstacle_radii = obstacle_radii

        # Set up the final state
        initial_state = self.current_state
        x, y = initial_state[0], initial_state[1]
        theta = initial_state[2]
        self._goal_state = np.concatenate([np.array([x, y, theta]) + relative_goal, np.array([0, 0, 0, 0])])

    def get_next_state(self, state, action):
        """ Simulate one step of nonlinear dynamics """
        h = self.time_step_duration 
        next_state = np.zeros_like(state)
        next_state[0] = state[0] + h * state[3] * np.cos(state[2])  # xpos 
        next_state[1] = state[1] + h * state[3] * np.sin(state[2])  # ypos
        next_state[2] = state[2] + h * state[3] * state[4]          # theta
        next_state[3] = state[3] + h * state[5]                     # velocity
        next_state[4] = state[4] + h * state[6]                     # kappa
        next_state[5] = state[5] + h * action[0]                    # accel
        next_state[6] = state[6] + h * action[1]                    # pinch
        return next_state

    def rollout_actions(self, state, actions):
        assert len(actions.shape) == 2 and actions.shape[1] == self.num_actions()
        assert len(state.shape) == 1 and state.shape[0] == self.num_states()
        num_time_steps = actions.shape[0]
        state_trajectory = np.zeros((num_time_steps+1, state.shape[0]))
        state_trajectory[0] = state
        for k in range(num_time_steps):
            state_trajectory[k+1] = self.get_next_state(state_trajectory[k], actions[k])
        return state_trajectory

    def _get_env_vars(self):
        """ Get a subset of MPC state variables from the environment 
        
        """
        theta_mpc = self.car.hull.angle + np.pi / 2
        vec1 = np.array(self.car.hull.linearVelocity) # Velocity as a vector
        vec2 = rotate_by_angle(np.array([1,0]), theta_mpc)
        velocity_mpc = np.dot(vec1, vec2)
        kappa_mpc = np.tan(self.car.wheels[0].joint.angle) / self.constants.ell

        x_env = (1/2)*(self.car.wheels[2].position[0]+self.car.wheels[3].position[0])
        y_env = (1/2)*(self.car.wheels[2].position[1]+self.car.wheels[3].position[1])
        x_mpc = x_env
        y_mpc = y_env

        return np.array([
            x_mpc, 
            y_mpc, 
            theta_mpc, 
            velocity_mpc, 
            kappa_mpc
        ])

    def take_action(self, action):
        """ Receive MPC action and feed it to the underlying environment 
        
        Expects np.ndarray of (jerk, juke)
        """
        next_state = self.get_next_state(self.current_state, action)
        
        # Get the env action from the MPC state and take it
        kappa, accel = next_state[4], next_state[5] 
        steering_action = -1*np.arctan(self.constants.ell * kappa)
        gas_action = (1/500) * accel  # Polo's magic constant
        brake_action = 0
        env_action = np.array([steering_action, gas_action, brake_action])
        _, reward, done, info = self.step(env_action)
        self._current_state = np.concatenate([self._get_env_vars(), next_state[5:7]])
        
        return self.current_state, reward, done, info
        