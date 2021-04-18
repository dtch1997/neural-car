import cvxpy as cp 
import numpy as np 
from typing import Dict

def curr(var: cp.Variable):
    return var[1:]
def prev(var: cp.Variable):
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
        self.parameters = AttrDict.from_dict({
            "time_step_magnitude": cp.Parameter(shape = (), value = self.time_step_magnitude),
            "prev_xpos": cp.Parameter(shape = num_time_steps+1),
            "prev_ypos": cp.Parameter(shape = num_time_steps+1),
            "prev_velocity": cp.Parameter(shape = num_time_steps+1),
            "prev_theta": cp.Parameter(shape = num_time_steps+1),
            "prev_kappa": cp.Parameter(shape = num_time_steps+1)
        })
        self.variables = AttrDict.from_dict({
            "xpos": cp.Variable(num_time_steps+1),
            "ypos": cp.Variable(num_time_steps+1),
            "velocity" : cp.Variable(num_time_steps+1),
            "theta" : cp.Variable(num_time_steps+1),
            "kappa" : cp.Variable(num_time_steps+1),
            "jerk" : cp.Variable(num_time_steps),
            "pinch" : cp.Variable(num_time_steps)
        })

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
        # Previous state trajectories are stored in self.parameters
        return []

    def solve(self, inputs: Dict[str, np.ndarray] = {}):
        for key, value_np in inputs:
            assert key in self.variables
            self.variables[key].value = value_np
        problem = cp.Problem(self.objective, self.constraints)
        problem.solve()

if __name__ == "__main__":
    solver = SCPSolver(100, 10)
    solver.solve()
    print(solver.variables.jerk.value)

