import cvxpy as cp
import numpy as np
from typing import Dict
import env as car_env

from typing import List
from copy import deepcopy
from dataclasses import dataclass

""" Slicing convenience functions """
def nxt(var: cp.Variable):
    return var[1:]
def curr(var: cp.Variable):
    return var[:-1]

@dataclass
class Obstacle:
    xpos: float 
    ypos: float
    radius: float

    @property
    def position(self) -> np.ndarray:
        return np.array([self.xpos, self.ypos])

class AttrDict(Dict):
    """ Dictionary that also lets you get the entries as properties """
    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        setattr(self, key, value)
    @staticmethod
    def from_dict(dict):
        attrdict =  AttrDict()
        for key, value in dict.items():
            attrdict[key] = value # Calls __setitem_
        return attrdict

class SCPSolver:
    def __init__(self, num_time_steps, duration,
            initial_position: np.ndarray,
            final_position: np.ndarray,
            max_jerk: float,
            max_juke: float,
            max_velocity: float,
            max_kappa: float,
            max_deviation_from_reference: float,
            obstacles: List[Obstacle] = []
        ):
        """
        :param obstacles: A List of (xpos, ypos, radius)
        """
        self.num_time_steps = num_time_steps
        self.duration = duration
        time_step_magnitude = duration / num_time_steps
        # Parameter semantics:
        #   State variables:
        #   - self.variables.xpos[0] is the initial state
        #   Inputs:
        #   - state[t+1] = f(state[t], input[t])

        self.constants = AttrDict.from_dict({
            "time_step_magnitude": cp.Constant(time_step_magnitude),
            "initial_position": cp.Constant(initial_position),
            "final_position": cp.Constant(final_position),
            "max_jerk": cp.Constant(max_jerk),
            "max_juke": cp.Constant(max_juke),
            "max_velocity": cp.Constant(max_velocity),
            "max_kappa": cp.Constant(max_kappa),
            "max_deviation_from_reference": cp.Constant(max_deviation_from_reference),
        })
        self.obstacles = obstacles

        zeros = np.zeros(num_time_steps+1)
        self.parameters = AttrDict.from_dict({
            "prev_xpos": cp.Parameter(shape = num_time_steps+1, value = zeros.copy()),
            "prev_ypos": cp.Parameter(shape = num_time_steps+1, value = zeros.copy()),
            "prev_velocity": cp.Parameter(shape = num_time_steps+1, value = zeros.copy()),
            "prev_theta": cp.Parameter(shape = num_time_steps+1, value = zeros.copy()),
            "prev_kappa": cp.Parameter(shape = num_time_steps+1, value = zeros.copy()),
            "prev_accel": cp.Parameter(shape = num_time_steps+1, value = zeros.copy()),
            "prev_pinch": cp.Parameter(shape = num_time_steps+1, value = zeros.copy())
        })
        self.variables = AttrDict.from_dict({
            "xpos": cp.Variable(num_time_steps+1),
            "ypos": cp.Variable(num_time_steps+1),
            "velocity" : cp.Variable(num_time_steps+1),
            "theta" : cp.Variable(num_time_steps+1),
            "kappa" : cp.Variable(num_time_steps+1),
            "accel": cp.Variable(num_time_steps+1),
            "pinch" : cp.Variable(num_time_steps+1),
            "jerk" : cp.Variable(num_time_steps),
            "juke" : cp.Variable(num_time_steps),
        })

        self.problem = cp.Problem(self.objective, self.constraints)

    @property
    def input(self):
        """ Get all the variables that encode the input to the system """
        return cp.vstack([
            self.variables.jerk,
            self.variables.juke
        ])

    @property 
    def position(self):
        return cp.vstack([
            self.variables.xpos,
            self.variables.ypos
        ])

    @property
    def state(self):
        """ Get all the variables that encode the state of the system """
        return cp.vstack([
            self.variables.xpos,
            self.variables.ypos,
            self.variables.velocity,
            self.variables.theta,
            self.variables.kappa,
            self.variables.accel,
            self.variables.pinch
        ])

    @property
    def objective(self):
        input = cp.vstack([self.variables.jerk, self.variables.juke])
        assert input.shape == (2, self.num_time_steps)
        input_norm = cp.norm(input, axis=0)
        assert input_norm.shape == (self.num_time_steps,)
        return cp.Minimize(cp.sum(input_norm) + cp.norm(self.position[:,-1] - self.constants.final_position))

    @property
    def constraints(self):
        xpos = self.variables.xpos
        ypos = self.variables.ypos
        veloc = self.variables.velocity
        theta = self.variables.theta
        kappa = self.variables.kappa
        accel = self.variables.accel
        pinch = self.variables.pinch
        jerk = self.variables.jerk
        juke = self.variables.juke

        # Previous estimates of state trajectories are stored in self.parameters
        prev_xpos = self.parameters.prev_xpos
        prev_ypos = self.parameters.prev_ypos
        prev_veloc = self.parameters.prev_velocity
        prev_theta = self.parameters.prev_theta
        prev_kappa = self.parameters.prev_kappa
        prev_accel = self.parameters.prev_accel
        prev_pinch = self.parameters.prev_pinch

        h = self.constants.time_step_magnitude
        r = self.constants.max_deviation_from_reference
        delta_theta = curr(theta) - curr(prev_theta) # 0i - 0i^{(k)}
        constraints = []

        """ Add the geometric constraints """
        constraints += [
            nxt(xpos) == curr(xpos) + h * (
                    cp.multiply(curr(veloc), np.cos(curr(prev_theta).value))
                    - cp.multiply(cp.multiply(curr(prev_veloc), np.sin(curr(prev_theta).value)), delta_theta)
                ),
            nxt(ypos) == curr(ypos) + h * (
                    cp.multiply(curr(veloc), np.sin(curr(prev_theta).value))
                    - cp.multiply(cp.multiply(curr(prev_veloc), np.cos(curr(prev_theta).value)), delta_theta)
                ),
            nxt(theta) == curr(theta) + h * (
                    cp.multiply(curr(veloc), curr(prev_kappa.value))
                    + cp.multiply(curr(prev_veloc), curr(kappa) - curr(prev_kappa))
                ),
            nxt(kappa) == curr(kappa) + h * curr(pinch),
            nxt(accel) == curr(accel) + h * jerk,
            nxt(pinch) == curr(pinch) + h * juke,
            xpos[0] == self.constants.initial_position[0],
            ypos[0] == self.constants.initial_position[1],
            #xpos[-1] == self.constants.final_position[0],
            #ypos[-1] == self.constants.final_position[1],
            cp.norm(jerk, p=np.inf) <= self.constants.max_jerk,
            cp.norm(juke, p=np.inf) <= self.constants.max_juke,
            cp.norm(veloc, p=np.inf) <= self.constants.max_velocity,
            cp.norm(kappa, p=np.inf) <= self.constants.max_kappa
        ]
        # TODO: Add the obstacle avoidance constraints 
        constraints += [
            
        ]

        # Add the max deviation from reference constraint
        constraints += [

        ]

        return constraints

    def solve(self) -> float:
        optval = self.problem.solve(solver = cp.SCS)
        # Initialize the parameters to the values found by solve
        # This way, the next time we call 'solve' we'll already have the right values
        if self.problem.status in ["infeasible", "unbounded"]:
            raise Exception(f"The problem was {self.problem.status}")
        for param_key in self.parameters.keys():
            var_key = param_key[5:] # get the corresponding variable
            self.parameters[param_key].value = self.variables[var_key].value
        return optval

def updateInitialPosition(env, solver):
    x = (1/2)*(env.car.wheels[2].position[0]+env.car.wheels[3].position[0])
    y = (1/2)*(env.car.wheels[2].position[1]+env.car.wheels[3].position[1])
    init_pos = np.array([x,y])
    solver.constants.initial_position = cp.Constant(init_pos)

if __name__ == "__main__":
    
    env = car_env.CarRacing(
            allow_reverse=True,
            grayscale=1,
            show_info_panel=1,
            discretize_actions=None,
            num_obstacles=100,
            num_tracks=1,
            num_lanes=1,
            num_lanes_changes=4,
            max_time_out=0,
            frames_per_state=4)

    env.reset()  # Put the car at the starting position

    # Obtain initial state information
    x = (1/2)*(env.car.wheels[2].position[0]+env.car.wheels[3].position[0])
    y = (1/2)*(env.car.wheels[2].position[1]+env.car.wheels[3].position[1])
    theta = env.car.hull.angle 
    velocity = env.car.hull.linearVelocity 
    ell = 80+82 # Obtained in neural car dynamics global variables
    kappa = np.tan(env.car.wheels[0].angle)/ell 

    # Default initial position
    init_pos = np.array([x,y])
    # Default final position 
    final_pos = ([40, 40])
    
    # Initialize to very high value until further notice
    very_high_value = 10**(14)
    max_jerk = very_high_value
    max_juke = very_high_value
    max_velocity = very_high_value
    max_kappa = very_high_value
    max_deviation_from_reference = very_high_value

    solver = SCPSolver(100, 10, init_pos, final_pos,
            max_jerk, max_juke, max_velocity, max_kappa, max_deviation_from_reference)

    diff = np.inf #initialize to unreasonable value to overwrite in loop
    epsilon = 0.01 #tolerance for convergence of the solution
    prevCost = -1*np.inf

    action = np.zeros(3)

    for _ in range(1000):
      env.render()
      updateInitialPosition(env, solver)
      while abs(diff) > epsilon:
          print("before solve")
          optval = solver.solve()
          #print('opt :',optval) #monitor
          print("after solve")
          diff = optval - prevCost
          #xt = deepcopy(x.value) #copy of state trajectory
          
      # Obtain the chosen action given the MPC solve
      kappa = solver.variables.kappa[0].value
      action[0] = np.arctan(ell*kappa) # steering action
      SIZE = 0.02
      mass = 1000000*SIZE*SIZE # friction ~= mass (as stated in dynamics)
      acc = solver.variables.accel[0].value
      action[1] = mass*acc # gas action
      action[2] = 0 # brake action - not used for our purposes
      
      # Step through the environment
      observation, reward, done, info = env.step(action)

      if done:
        observation = env.reset()
    env.close