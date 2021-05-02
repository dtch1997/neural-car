import numpy as np

from .car_racing import CarRacing, FPS
from .dynamics import SIZE
from src.utils import AttrDict
from typing import *



def rotate_by_angle(vec, th):
    """ Rotate a 2D vector by angle

    :param vec: np.array of shape (2,)
    :param th: angle in radians
    """
    M = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
    return M @ vec

class Environment(CarRacing):
    """ A wrapper around the CarRacing environment 
    
    Provides APIs for: 
    1. Extracting state variables
    2. Converting inputs back to openAI gym controls
    """ 
    constants = AttrDict.from_dict({
        "size": SIZE, 
        "ell": SIZE * (80 + 82),
        "fps": FPS # Number of frames per second in the simulation
    })

    state_variable_names = ("xpos", "ypos", "velocity", "theta")
    input_variable_names = ("kappa", "accel")

    @staticmethod
    def num_states():
        return len(Environment.state_variable_names)
    
    @staticmethod 
    def num_inputs():
        return len(Environment.input_variable_names)

    def reset_history(self):
        num_state_vars = len(self.state_variable_names)
        # Note: DO NOT use this variable directly outside of this class
        # Use Environment.state_history instead 
        self._state_history = np.zeros([32, num_state_vars])
        self._num_records = 0

    @property
    def state_history(self):
        return self.state_history[:self._num_records]

    def record_state(self):
        """ Automatically record the current state """
        state = self.get_state()
        for var_idx, var_name in enumerate(self.state_variable_names):
            self.state_history[self.num_logs, var_idx] = state[var_name]
        self._num_records += 1 
        # Expand the state history if we've run out of space 
        if self._num_records >= self._state_history.shape[0]:
            self._state_history = np.concatenate([self._state_history, np.zeros_like(self._state_history)], axis=-1)
    
    def get_state(self) -> AttrDict[str, float]:
        """ Get the current state from underlying environment """
        theta_mpc = env.car.hull.angle + np.pi / 2
        vec1 = np.array(env.car.hull.linearVelocity) # Velocity as a vector
        vec2 = rotate_by_angle(np.array([1,0]), theta_mpc)
        velocity_mpc = np.dot(vec1, vec2)
        kappa_mpc = np.tan(env.car.wheels[0].joint.angle) / self.constants.ell

        x_env = (1/2)*(env.car.wheels[2].position[0]+env.car.wheels[3].position[0])
        y_env = (1/2)*(env.car.wheels[2].position[1]+env.car.wheels[3].position[1])
        x_mpc = x_env
        y_mpc = y_env

        return AttrDict.from_dict({
            "xpos": x_mpc,
            "ypos": y_mpc,
            "velocity": velocity_mpc,
            "theta": theta_mpc
        })

    def step(self, action: AttrDict):
        """ Receive MPC action and feed it to the underlying environment """
        assert set(action.keys()) == set(self.input_variable_names)
        kappa, accel = action.kappa, action.accel 
        action = None 

        steering_action = -1*np.arctan(self.constants.ell * kappa)
        gas_action = (1/500) * accel  # Polo's magic constant
        brake_action = 0
        env_action = np.array([steering_action, gas_action, brake_action])
        self.record_state()
        return super(CarRacing, self).step(env_action)
