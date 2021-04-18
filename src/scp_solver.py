import cvxpy as cp 
import numpy as np 
from typing import Dict


""" Slicing convenience functions """
def nxt(var: cp.Variable):
    return var[1:]
def curr(var: cp.Variable):
    return var[:-1]

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
    def __init__(self, num_time_steps, duration):
        self.num_time_steps = num_time_steps
        self.duration = duration
        self.time_step_magnitude = duration / num_time_steps 

        # Parameter semantics: 
        #   State variables:
        #   - self.variables.xpos[0] is the initial state
        #   Inputs:
        #   - state[t+1] = f(state[t], input[t])
        zeros = np.zeros(num_time_steps+1)
        self.parameters = AttrDict.from_dict({
            "prev_xpos": cp.Parameter(shape = num_time_steps+1, value = zeros.copy()),
            "prev_ypos": cp.Parameter(shape = num_time_steps+1, value = zeros.copy()),
            "prev_velocity": cp.Parameter(shape = num_time_steps+1, value = zeros.copy()),
            "prev_theta": cp.Parameter(shape = num_time_steps+1, value = zeros.copy()),
            "prev_kappa": cp.Parameter(shape = num_time_steps+1, value = zeros.copy()),
            "prev_acceleraton": cp.Parameter(shape = num_time_steps+1, value = zeros.copy()),
            "prev_juke": cp.Parameter(shape = num_time_steps+1, value = zeros.copy())
        })
        self.variables = AttrDict.from_dict({
            "xpos": cp.Variable(num_time_steps+1),
            "ypos": cp.Variable(num_time_steps+1),
            "velocity" : cp.Variable(num_time_steps+1),
            "theta" : cp.Variable(num_time_steps+1),
            "kappa" : cp.Variable(num_time_steps+1),
            "acceleration": cp.Variable(num_time_steps+1), 
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
            self.variables.pinch
        ])

    @property 
    def state(self):
        """ Get all the variables that encode the state of the system """
        return cp.vstack([
            self.variables.xpos, 
            self.variables.ypos, 
            self.variables.velocity,
            self.variables.theta, 
            self.variables.kappa
        ])

    @property
    def objective(self):
        input = cp.vstack([self.variables.jerk, self.variables.pinch])
        assert input.shape == (2, self.num_time_steps)
        input_norm = cp.norm(input, axis=0)
        assert input_norm.shape == (self.num_time_steps,)
        return cp.Minimize(cp.sum(input_norm))

    @property
    def constraints(self):        
        xpos = self.variables.xpos 
        ypos = self.variables.ypos
        veloc = self.variables.velocity
        theta = self.variables.theta
        kappa = self.variables.kappa 
        jerk = self.variables.jerk
        pinch = self.variables.pinch

        # Previous estimates of state trajectories are stored in self.parameters
        prev_xpos = self.parameters.prev_xpos
        prev_ypos = self.parameters.prev_ypos 
        prev_veloc = self.parameters.prev_velocity
        prev_theta = self.parameters.prev_theta
        prev_kappa = self.parameters.prev_kappa

        h = self.num_time_steps 
        delta_theta = curr(theta) - curr(prev_theta) # 0i - 0i^{(k)}

        return [
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
                    + cp_multiply(curr(prev_veloc), curr(kappa) - curr(prev_kappa)))
                )
        ]

    def solve(self) -> float:
        optval = self.problem.solve()
        # Initialize the parameters to the values found by solve
        # This way, the next time we call 'solve' we'll already have the right values
        for param_key in self.parameters.keys():
            var_key = param_key[5:] # get the corresponding variable
            self.parameters[param_key].value = self.variables[var_key].value
        return optval

if __name__ == "__main__":
    solver = SCPSolver(100, 10)
    solver.solve()
    print(solver.variables.xpos.value)
    print(solver.parameters.prev_xpos.value)

