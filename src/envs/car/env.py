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
    def state_history(self):
        return self.state_history[:self._num_records]

    @property 
    def current_state(self):
        return self._current_state.copy()

    def reset(self):
        super(Environment, self).reset()
        # When resetting, the acceleration and pinch are always zero
        self._current_state = np.concatenate([self._get_env_vars(), np.zeros(2)])

    def reset_history(self):
        num_state_vars = len(self.state_variable_names)
        # Note: DO NOT use this variable directly outside of this class
        # Use Environment.state_history instead 
        self._state_history = np.zeros([32, num_state_vars])
        self._num_records = 0

    def record_state(self):
        """ Automatically record the current state """
        state = self.current_state
        for var_idx, var_name in enumerate(self.state_variable_names):
            self.state_history[self._num_records, var_idx] = state[var_name]
        self._num_records += 1 
        # Expand the state history if we've run out of space 
        if self._num_records >= self._state_history.shape[0]:
            self._state_history = np.concatenate([self._state_history, np.zeros_like(self._state_history)], axis=-1)

    def get_next_state(self, state, action):
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
        """ Get a subset of state variables from the environment """
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
        kappa, accel = next_state[4], next_state[5] 

        steering_action = -1*np.arctan(self.constants.ell * kappa)
        gas_action = (1/500) * accel  # Polo's magic constant
        brake_action = 0
        env_action = np.array([steering_action, gas_action, brake_action])
        print(env_action)
        # self.record_state()
        self.step(env_action)
        self._current_state = next_state

        return self.current_state
