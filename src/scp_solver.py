#import cvxpy as cp
import cyipopt
import numpy as np
import env as car_env

from typing import List, Dict
from copy import deepcopy
from dataclasses import dataclass

""" Slicing convenience functions """
def nxt(var: np.array):
    return var[1:]
def curr(var: np.array):
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

<<<<<<< Updated upstream
class SCPSolver:
    def __init__(self, num_time_steps, duration,
            initial_position: np.ndarray,
=======
class NLPSolver:
    """ A sequential convex programming solver for the CarRacing OpenAI gym environment

    Usage:

    solver = SCPSolver(...)
    for time_step in range(MAX_TIME_STEPS):

    """

    state_variable_names = ["xpos", "ypos", "velocity", "theta", "kappa", "accel", "pinch"]
    input_variable_names = ["jerk", "juke"]

    def __init__(self,
            num_time_steps: float,
            duration: float,
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
=======

>>>>>>> Stashed changes
        # Parameter semantics:
        #   State variables:
        #   - self.variables.xpos[0] is the initial state
        #   Inputs:
        #   - state[t+1] = f(state[t], input[t])

        self.constants = AttrDict.from_dict({
<<<<<<< Updated upstream
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

    def linear_init(self, init_pos, final_pos, velocity, theta, kappa):
        self.parameters.prev_xpos = np.linspace(init_pos[0], final_pos[0], self.num_time_steps+1)
        self.parameters.prev_ypos = np.linspace(init_pos[1], final_pos[1], self.num_time_steps+1)
        self.parameters.prev_velocity = velocity*np.ones(self.num_time_steps+1)
        self.parameters.prev_theta = theta*np.ones(self.num_time_steps+1)
        self.parameters.prev_kappa = kappa*np.ones(self.num_time_steps+1)
=======
            "time_step_magnitude": time_step_magnitude,
            "final_position": final_position,
            "max_jerk": max_jerk,
            "max_juke": max_juke,
            "max_velocity": max_velocity,
            "max_kappa": max_kappa,
            "max_accel": max_accel,
            "max_pinch": max_pinch,
            "max_theta": max_theta,
            "max_deviation_from_reference": max_deviation_from_reference,
        })
        self.obstacles = obstacles

        self.previous_trajectory = AttrDict.from_dict({
            state_variable_name: np.zeros(num_time_steps+1) \
                for state_variable_name in self.state_variable_names
        })
        # Current state
        self.current_state = AttrDict.from_dict({
            state_variable_name: np.zeros(1) \
                for state_variable_name in self.state_variable_names
        })
        self.variables = AttrDict.from_dict({
            # Note: Idiomatic way to combine two dictionaries
            **{state_variable_name: np.zeros(num_time_steps+1) \
                for state_variable_name in self.state_variable_names},
            **{input_variable_name: np.zeros(num_time_steps) \
                for input_variable_name in self.input_variable_names}
        })

    def update_state(self, values: Dict[str, float], trajectory_init = "zero"):
        """ Update the current state of the car in the solver

        Also initializes a feasible trajectory from that state
        By default, this is the trajectory obtained by having zero input

        Usage:
            solver.update_state({
                "xpos": 23.4,
                "ypos": 14.5,
                ...
                "accel": 0.1,
                "pinch: 1.2
            })
        """
        for key, value in values.items():
            assert key in self.state_variable_names, "Invalid state variable entered"
            if value is None:
                value = self.previous_trajectory[key][1]
            assert value is not None
            self.current_state[key] = value

        if trajectory_init == "zero":
            self._init_trajectory_zero()
        else:
            raise ValueError(f"Trajectory initializatoin {trajectory_init} not recognized")

    def _init_trajectory_zero(self):
        """ Initialize the previous trajectory to the trajectory defined by zero input for all time

        I.e. car moves with fixed constant velocity
        """
        xpos = self.current_state.xpos
        ypos = self.current_state.ypos
        veloc = self.current_state.velocity
        theta = self.current_state.theta
        #h = self.constants.time_step_magnitude.value
        h = self.constants.time_step_magnitude
        # TODO: Ask polo to check this
        vx, vy = veloc * np.cos(theta), veloc * np.sin(theta)
        
        self.previous_trajectory.xpos = xpos + vx * h * np.arange(self.num_time_steps+1)
        self.previous_trajectory.ypos = ypos + vy * h * np.arange(self.num_time_steps+1)
        self.previous_trajectory.velocity = veloc * np.ones(self.num_time_steps+1)
        self.previous_trajectory.theta = theta * np.ones(self.num_time_steps+1)
        self.previous_trajectory.kappa = np.zeros(self.num_time_steps+1)
        self.previous_trajectory.accel = np.zeros(self.num_time_steps+1)
        self.previous_trajectory.pinch = np.zeros(self.num_time_steps+1)

    def establish_variables(self, x):
        increment = self.num_time_steps + 1
        self.variables.xpos = x[increment*0:increment*1]
        self.variables.ypos = x[increment*1:increment*2]
        self.variables.velocity = x[increment*2:increment*3]
        self.variables.theta = x[increment*3:increment*4]
        self.variables.kappa = x[increment*4:increment*5]
        self.variables.accel = x[increment*5:increment*6]
        self.variables.pinch = x[increment*6:increment*7]
        increment2 = self.num_time_steps
        self.variables.jerk = x[increment*7:increment*7+increment2]
        self.variables.juke = x[increment*7+increment2:increment*7+2*increment2]
>>>>>>> Stashed changes

    @property
    def input(self):
        """ Get all the variables that encode the input to the system """
        return np.vstack([
            self.variables.jerk,
            self.variables.juke
        ])

    @property
    def position(self):
        return np.vstack([
            self.variables.xpos,
            self.variables.ypos
        ])

    @property
    def state(self):
        """ Get all the variables that encode the state of the system """
        return np.vstack([
            self.variables.xpos,
            self.variables.ypos,
            self.variables.velocity,
            self.variables.theta,
            self.variables.kappa,
            self.variables.accel,
            self.variables.pinch
        ])

    def objective(self, x):
        self.establish_variables(x)
        input = np.vstack([self.variables.jerk, self.variables.juke])
        assert input.shape == (2, self.num_time_steps)
<<<<<<< Updated upstream
        input_norm = cp.norm(input, axis=0)
        assert input_norm.shape == (self.num_time_steps,)
        return cp.Minimize(cp.sum(input_norm) + cp.norm(self.position[:,-1] - self.constants.final_position))
=======
        input_norm_sq = np.linalg.norm(input, axis=0)**2
        assert input_norm_sq.shape == (self.num_time_steps,)
        return np.sum(input_norm_sq) \
            + np.linalg.norm(self.position[:,-1] - self.constants.final_position, 2)
>>>>>>> Stashed changes

    def gradient(self, x):
        self.establish_variables(x)
        
        # Establish the indices of x where the various variables are extracted
        increment = self.num_time_steps + 1
        xpos_idx = increment*0
        ypos_idx = increment*1
        velocity_idx = increment*2
        theta_idx = increment*3
        kappa_idx = increment*4
        accel_idx = increment*5
        pinch_idx = increment*6
        increment2 = self.num_time_steps
        jerk_idx = increment*7
        juke_idx = increment*7+increment2
        
        g = np.zeros_like(x)
        
        # Gradient component due to np.sum(input_norm_sq)
        g[jerk_idx:juke_idx] = 2*self.variables.jerk
        g[juke_idx:] = 2*self.variables.juke
        
        # Gradient component due to final position constraint
        
        # The final x position
        g[ypos_idx-1] = 2 * (self.variables.xpos[-1]-self.constants.final_position[0]) / np.linalg.norm(self.position[:,-1] - self.constants.final_position, 2)
        # The final y position
        g[velocity_idx-1] = 2 * (self.variables.ypos[-1]-self.constants.final_position[1]) / np.linalg.norm(self.position[:,-1] - self.constants.final_position, 2)
        # The final velocity position
        g[theta_idx-1] = 2 * (self.variables.velocity[-1]-self.constants.final_position[2]) / np.linalg.norm(self.position[:,-1] - self.constants.final_position, 2)
        # The final theta position
        g[kappa_idx-1] = 2 * (self.variables.theta[-1]-self.constants.final_position[3]) / np.linalg.norm(self.position[:,-1] - self.constants.final_position, 2)

        return g

    def jacobian(self, x):
        self.establish_variables(x)
        
        # Establish the indices of x where the various variables are extracted
        increment = self.num_time_steps + 1
        xpos_idx = increment*0
        ypos_idx = increment*1
        velocity_idx = increment*2
        theta_idx = increment*3
        kappa_idx = increment*4
        accel_idx = increment*5
        pinch_idx = increment*6
        increment2 = self.num_time_steps
        jerk_idx = increment*7
        juke_idx = increment*7+increment2
        
        J = np.zeros((x.shape[0], 7*increment+2*increment2))
        
        h = self.constants.time_step_magnitude
        
        # X-constraint part of Jacobian
        for k in range(self.num_time_steps):
            J[xpos_idx+k,xpos_idx+k] = 1
            J[xpos_idx+k,xpos_idx+k+1] = -1
            J[xpos_idx+k,velocity_idx+k] = h*np.cos(self.variables.theta[k])
            J[xpos_idx+k,theta_idx+k] = -h*self.variables.velocity[k]*np.sin(self.variables.theta[k])
        
        # Y-constraint part of Jacobian
        for k in range(self.num_time_steps):
            J[ypos_idx+k,ypos_idx+k] = 1
            J[ypos_idx+k,ypos_idx+k+1] = -1
            J[ypos_idx+k,velocity_idx+k] = h*np.sin(self.variables.theta[k])
            J[ypos_idx+k,theta_idx+k] = h*self.variables.velocity[k]*np.cos(self.variables.theta[k])
        
        # Velocity-constraint part of Jacobian
        for k in range(self.num_time_steps):
            J[velocity_idx+k,velocity_idx+k] = 1
            J[velocity_idx+k,velocity_idx+k+1] = -1
            J[velocity_idx+k,accel_idx+k] = h
        
        # Theta-constraint part of Jacobian
        for k in range(self.num_time_steps):
            J[theta_idx+k,theta_idx+k] = 1
            J[theta_idx+k,theta_idx+k+1] = -1
            J[theta_idx+k,velocity_idx+k] = h*self.variables.kappa[k]
            J[theta_idx+k,kappa_idx+k] = h*self.variables.velocity[k]
        
        # Kappa-constraint part of Jacobian
        for k in range(self.num_time_steps):
            J[kappa_idx+k,kappa_idx+k] = 1
            J[kappa_idx+k,kappa_idx+k+1] = -1
            J[kappa_idx+k,pinch_idx+k] = h
            
        # Pinch-constraint part of Jacobian
        for k in range(self.num_time_steps):
            J[pinch_idx+k,pinch_idx+k] = 1
            J[pinch_idx+k,pinch_idx+k+1] = -1
            J[pinch_idx+k,juke_idx+k] = h
            
        # Accel-constraint part of Jacobian
        for k in range(self.num_time_steps):
            J[accel_idx+k,accel_idx+k] = 1
            J[accel_idx+k,accel_idx+k+1] = -1
            J[accel_idx+k,jerk_idx+k] = h
        
        return J

    def constraints(self, x):
        self.establish_variables(x)
        
        xpos = self.variables.xpos
        ypos = self.variables.ypos
        veloc = self.variables.velocity
        theta = self.variables.theta
        kappa = self.variables.kappa
        accel = self.variables.accel
        pinch = self.variables.pinch
        jerk = self.variables.jerk
        juke = self.variables.juke

<<<<<<< Updated upstream
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
                    + cp.multiply(cp.multiply(curr(prev_veloc), np.cos(curr(prev_theta).value)), delta_theta)
                ),
            nxt(theta) == curr(theta) + h * (
                    cp.multiply(curr(veloc), curr(prev_kappa.value))
                    + cp.multiply(curr(prev_veloc), curr(kappa) - curr(prev_kappa))
                ),
            nxt(veloc) == curr(veloc) + h * curr(accel),
            nxt(kappa) == curr(kappa) + h * curr(pinch),
            nxt(accel) == curr(accel) + h * jerk,
            nxt(pinch) == curr(pinch) + h * juke,
            xpos[0] == self.constants.initial_position[0],
            ypos[0] == self.constants.initial_position[1],
            #veloc[-1] == 0,
            #xpos[-1] == self.constants.final_position[0],
            #ypos[-1] == self.constants.final_position[1],
            cp.norm(jerk, p=np.inf) <= self.constants.max_jerk,
            cp.norm(juke, p=np.inf) <= self.constants.max_juke,
            cp.norm(veloc, p=np.inf) <= self.constants.max_velocity,
            cp.norm(kappa, p=np.inf) <= self.constants.max_kappa,
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

        self.linear_init(
            self.constants.initial_position.value,
            self.constants.final_position.value,
            self.variables.velocity.value[1],
            self.variables.theta.value[1],
            self.variables.kappa.value[1]
        )
        return optval

def rotateByAngle(vec, th):
=======
        h = self.constants.time_step_magnitude
        r = self.constants.max_deviation_from_reference

        constraints = np.concatenate((
            -nxt(xpos) + curr(xpos) + h * curr(veloc)*np.cos(curr(theta)),
            -nxt(ypos) + curr(ypos) + h * curr(veloc)*np.sin(curr(theta)),
            -nxt(veloc) + curr(veloc) + h * curr(accel),
            -nxt(theta) + curr(theta) + h * (curr(veloc)*curr(kappa)),
            -nxt(kappa) + curr(kappa) + h * curr(pinch),
            -nxt(accel) + curr(accel) + h * jerk,
            -nxt(pinch) + curr(pinch) + h * juke
        ))

        return constraints

def rotate_by_angle(vec, th):
>>>>>>> Stashed changes
    M = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
    return M@vec

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
    vec1 = np.array(env.car.hull.linearVelocity) # Velocity as a vector
    vec2 = rotateByAngle(np.array([1,0]), theta)
    dot_prod = np.dot(vec1, vec2)
    velocity = np.linalg.norm(vec1,2) if dot_prod > 0 else -np.linalg.norm(vec1,2)
    print(velocity)
    ell = 80+82 # Obtained in neural car dynamics global variables
    kappa = np.tan(env.car.wheels[0].angle)/ell

    # Default initial position
    init_pos = np.array([x,y])
    # Default final position
    final_pos = np.array([x+10, y])

    # Initialize to very high value until further notice
    very_high_value = 10**(14)
    max_jerk = very_high_value
    max_juke = very_high_value
    max_velocity = 10
    max_kappa = 0.2
    max_deviation_from_reference = very_high_value
    epsilon = 0.01 #tolerance for convergence of the solution
    action = np.zeros(3)
<<<<<<< Updated upstream

    solver = SCPSolver(100, 2, init_pos, final_pos,
        max_jerk, max_juke, max_velocity, max_kappa, max_deviation_from_reference)
    # solver.linearInit(init_pos, final_pos, velocity, theta, kappa)

    for _ in range(1000):
        # Linearly interpolate init and final pos
        updateInitialPosition(env, solver)

        diff = np.inf #initialize to unreasonable value to overwrite in loop
        prevCost = -1*np.inf
        env.render()

        while abs(diff) > epsilon:
            optval = solver.solve()
            #print('opt :',optval) #monitor
            diff = optval - prevCost
            prevCost = optval
            print(solver.problem.status, optval, diff)
            #xt = deepcopy(x.value) #copy of state trajectory

        # Obtain the chosen action given the MPC solve
        kappa = solver.variables.kappa[0].value
        action[0] = np.arctan(ell*kappa) # steering action
        SIZE = 0.02
        mass = 1000000*SIZE*SIZE # friction ~= mass (as stated in dynamics)
        acc = solver.variables.accel[0].value
        action[1] = mass*acc # gas action
=======
    max_theta = np.pi

    solver = NLPSolver(
        num_time_steps = 100,
        duration = 2,
        final_position = final_position,
        max_jerk = max_jerk,
        max_juke = max_juke,
        max_velocity = max_velocity,
        max_kappa = max_kappa,
        max_accel = max_accel,
        max_pinch = max_pinch,
        max_theta = max_theta,
        max_deviation_from_reference = max_deviation_from_reference
    )
    
    solver.update_state(initial_state)

    n = 7*(solver.num_time_steps+1)+2*(solver.num_time_steps)
    m = 7*(solver.num_time_steps)

    increment = solver.num_time_steps + 1
    xpos_idx = increment*0
    ypos_idx = increment*1
    velocity_idx = increment*2
    theta_idx = increment*3
    kappa_idx = increment*4
    accel_idx = increment*5
    pinch_idx = increment*6
    increment2 = solver.num_time_steps
    jerk_idx = increment*7
    juke_idx = increment*7+increment2


    lb = -np.inf*np.ones(n)
    lb[xpos_idx] = solver.current_state["xpos"]
    lb[ypos_idx] = solver.current_state["ypos"]
    lb[velocity_idx] = solver.current_state["velocity"]
    lb[theta_idx] = solver.current_state["theta"]
    lb[kappa_idx] = solver.current_state["kappa"]
    lb[accel_idx] = solver.current_state["accel"]
    lb[pinch_idx] = solver.current_state["pinch"]
    
    ub = np.inf*np.ones(n)
    ub[xpos_idx] = solver.current_state["xpos"]
    ub[ypos_idx] = solver.current_state["ypos"]
    ub[velocity_idx] = solver.current_state["velocity"]
    ub[theta_idx] = solver.current_state["theta"]
    ub[kappa_idx] = solver.current_state["kappa"]
    ub[accel_idx] = solver.current_state["accel"]
    ub[pinch_idx] = solver.current_state["pinch"]

    lb[velocity_idx:theta_idx] = -solver.constants["max_velocity"]
    lb[theta_idx:kappa_idx] = -solver.constants["max_theta"]
    lb[kappa_idx:accel_idx] = -solver.constants["max_kappa"]
    lb[accel_idx:pinch_idx] = -solver.constants["max_accel"]
    lb[pinch_idx:jerk_idx] = -solver.constants["max_pinch"]
    lb[jerk_idx:juke_idx] = -solver.constants["max_jerk"]
    lb[juke_idx:] = -solver.constants["max_juke"]

    ub[velocity_idx:theta_idx] = solver.constants["max_velocity"]
    ub[theta_idx:kappa_idx] = solver.constants["max_theta"]
    ub[kappa_idx:accel_idx] = solver.constants["max_kappa"]
    ub[accel_idx:pinch_idx] = solver.constants["max_accel"]
    ub[pinch_idx:jerk_idx] = solver.constants["max_pinch"]
    ub[jerk_idx:juke_idx] = solver.constants["max_jerk"]
    ub[juke_idx:] = solver.constants["max_juke"]

    cl = np.zeros(m)
    cu = np.zeros(m)

    nlp = cyipopt.Problem(
       n=n,
       m=m,
       problem_obj=solver,
       lb=lb,
       ub=ub,
       cl=cl,
       cu=cu,
    ) 

    NUM_TIME_STEPS = 100
    actual_trajectory = np.zeros([NUM_TIME_STEPS, 7])
    first = True
    fig, ax = plt.subplots(2,3)

    derivative = np.zeros(100)
    prev_velocity = 0

    for _ in range(NUM_TIME_STEPS):
        env.render()

        x0 = np.concatenate((
            solver.previous_trajectory.xpos,
            solver.previous_trajectory.ypos,
            solver.previous_trajectory.velocity,
            solver.previous_trajectory.theta,
            solver.previous_trajectory.kappa,
            solver.previous_trajectory.accel,
            solver.previous_trajectory.pinch,
            np.zeros(solver.num_time_steps),
            np.zeros(solver.num_time_steps)
        ))
        x, info = nlp.solve(x0)

        if first:
            ax[0,0].scatter(solver.variables.xpos, solver.variables.ypos, c='black', label = 'Planned trajectory')
            ax[0,0].scatter(solver.current_state.xpos, solver.current_state.ypos, s=30, c='blue')
            #ax[0,0].scatter(solver.constants.final_position.value[0], solver.constants.final_position.value[1], s=30, c='red')
            ax[0,0].scatter(solver.constants.final_position[0], solver.constants.final_position[1], s=30, c='red')
            ax[0,1].plot(np.arange(solver.num_time_steps+1), solver.variables.velocity.value)
            ax[0,2].plot(np.arange(solver.num_time_steps+1), solver.variables.accel.value)
            ax[1,0].plot(np.arange(solver.num_time_steps+1), solver.variables.theta.value)
            ax[1,1].plot(np.arange(solver.num_time_steps+1), solver.variables.kappa.value)
            ax[1,2].plot(np.arange(solver.num_time_steps+1), np.arctan(ELL * solver.variables.kappa.value))
            print("theta below")
            print(solver.variables.theta)
            
            first = False
        
        # Obtain the chosen action given the MPC solve
        kappa = solver.variables.kappa[0]
        action[0] = np.arctan(ELL * kappa) / -0.4200316 # steering action, rescale
        mass = 1000000*SIZE*SIZE # friction ~= mass (as stated in dynamics)
        alpha = (1/43.77365112) # Magicccccc!
        acc = solver.variables.accel[0]
        action[1] = alpha*acc
>>>>>>> Stashed changes
        action[2] = 0 # brake action - not used for our purposes

        # Step through the environment
        observation, reward, done, info = env.step(action)

        if done:
            observation = env.reset()
    env.close
