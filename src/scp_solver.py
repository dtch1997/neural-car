#import cvxpy as cp
import cyipopt
import numpy as np
import env as car_env
import matplotlib.pyplot as plt

from typing import List, Dict
from copy import deepcopy
from dataclasses import dataclass

SIZE = 0.02
ELL = SIZE*(80 + 82) # Length of car; defined in neural-car-dynamics global variables

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
            initial_position: np.ndarray,
            final_position: np.ndarray,
            max_jerk: float,
            max_juke: float,
            max_velocity: float,
            max_kappa: float,
            max_accel: float,
            max_pinch: float,
            max_theta: float,
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
            "time_step_magnitude": time_step_magnitude,
            "initial_position": initial_position,
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
        self.current_state.xpos = initial_position[0]    
        self.current_state.xpos = initial_position[1]
        self.current_state.velocity = initial_position[2]  
        self.current_state.theta = initial_position[3]  
            
        self.variables = AttrDict.from_dict({
            # Note: Idiomatic way to combine two dictionaries
            **{state_variable_name: np.zeros(num_time_steps+1) \
                for state_variable_name in self.state_variable_names},
            **{input_variable_name: np.zeros(num_time_steps) \
                for input_variable_name in self.input_variable_names}
        })

    def shift_prev_trajectory(self):
        for state_variable_name in self.state_variable_names:
            self.prev_trajectory[state_variable_name] = np.concatenate([self.prev_trajectory[state_variable_name][1:], self.prev_trajectory[state_variable_name][-1][np.newaxis, :]], axis=0) 

    def update_state(self, values: Dict[str, float], rollout_states = None, rollout_actions = None, trajectory_init = "zero"):
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
        elif trajectory_init == "rollout":
            self._init_trajectory_rollout(rollout_states, rollout_actions)
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
        
        theta = np.arctan2((self.constants.final_position[1] - self.current_state.ypos),(self.constants.final_position[0] - self.current_state.xpos))
        
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

    def _init_trajectory_rollout(self, rollout_states, rollout_actions):
        """ Initialize the previous trajectory to the trajectory defined by zero input for all time

        I.e. car moves with fixed constant velocity
        """
        xpos = self.current_state.xpos
        ypos = self.current_state.ypos
        veloc = self.current_state.velocity
        theta = self.current_state.theta
        
        theta = np.arctan2((self.constants.final_position[1] - self.current_state.ypos),(self.constants.final_position[0] - self.current_state.xpos))
        
        #h = self.constants.time_step_magnitude.value
        h = self.constants.time_step_magnitude
        # TODO: Ask polo to check this
        vx, vy = veloc * np.cos(theta), veloc * np.sin(theta)
        
        self.previous_trajectory.xpos = rollout_states[:,0]
        self.previous_trajectory.ypos = rollout_states[:,1]
        self.previous_trajectory.velocity = rollout_states[:,3]
        self.previous_trajectory.theta = rollout_states[:,2]
        self.previous_trajectory.kappa = rollout_states[:,4]
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
        self.variables.juke = x[increment*7+increment2:]

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
            self.variables.ypos,
            self.variables.velocity,
            self.variables.theta
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
        input_norm_sq = np.linalg.norm(input, axis=0)**2
        assert input_norm_sq.shape == (self.num_time_steps,)
        return np.sum(input_norm_sq) \
            + np.linalg.norm(self.position[:,-1] - self.constants.final_position, 2)

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
        g[jerk_idx:juke_idx] += 2*self.variables.jerk
        g[juke_idx:] += 2*self.variables.juke
        
        # Gradient component due to final position constraint
        
        # The final x position
        g[ypos_idx-1] += (self.variables.xpos[-1]-self.constants.final_position[0]) / np.linalg.norm(self.position[:,-1] - self.constants.final_position, 2)
        # The final y position
        g[velocity_idx-1] += (self.variables.ypos[-1]-self.constants.final_position[1]) / np.linalg.norm(self.position[:,-1] - self.constants.final_position, 2)
        # The final velocity position
        g[theta_idx-1] += (self.variables.velocity[-1]-self.constants.final_position[2]) / np.linalg.norm(self.position[:,-1] - self.constants.final_position, 2)
        # The final theta position
        g[kappa_idx-1] += (self.variables.theta[-1]-self.constants.final_position[3]) / np.linalg.norm(self.position[:,-1] - self.constants.final_position, 2)

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
        
        return J.T

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
    M = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
    return M@vec

def get_current_state(env) -> Dict[str, float]:
    """ Get the current state from environment
    All variables should be converted to MPC coordinate system
    """
    theta_mpc = env.car.hull.angle + np.pi / 2
    vec1 = np.array(env.car.hull.linearVelocity) # Velocity as a vector
    vec2 = rotate_by_angle(np.array([1,0]), theta_mpc)
    dot_prod = np.dot(vec1, vec2)
    #velocity_mpc = np.linalg.norm(vec1,2) if dot_prod > 0 else -np.linalg.norm(vec1,2)
    # We should only count the forward velocity
    velocity_mpc = dot_prod
    kappa_mpc = np.tan(env.car.wheels[0].angle) / ELL

    x_env = (1/2)*(env.car.wheels[2].position[0]+env.car.wheels[3].position[0])
    y_env = (1/2)*(env.car.wheels[2].position[1]+env.car.wheels[3].position[1])
    #x_mpc = y_env
    #y_mpc = -x_env
    x_mpc = x_env
    y_mpc = y_env

    return {
        "xpos": x_mpc,
        "ypos": y_mpc,
        "velocity": velocity_mpc,
        "theta": theta_mpc,
        "kappa": kappa_mpc,
        "accel": None,
        "pinch": None
    }

def updateInitialPosition(env, solver):
    x = (1/2)*(env.car.wheels[2].position[0]+env.car.wheels[3].position[0])
    y = (1/2)*(env.car.wheels[2].position[1]+env.car.wheels[3].position[1])
    init_pos = np.array([x,y])
    solver.constants.initial_position = init_pos

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
    initial_state = get_current_state(env)
    initial_state['accel'] = 0
    initial_state['pinch'] = 0

    x, y = initial_state['xpos'], initial_state['ypos']
    theta = initial_state['theta']
    direction = np.array([np.cos(theta), np.sin(theta),0,0])
    orth_direction = np.array([*rotate_by_angle(direction[:2], np.pi/2),0,0])
    initial_position = np.array([x,y,0,theta])
    final_position = np.array([x,y,0,theta]) + 10 * direction + 10 * orth_direction

    print("Initial x: ", x)
    print("Initial y: ", y)
    print("Final x: ", final_position[0])
    print("Final y: ", final_position[1])

    # Initialize to very high value until further notice
    very_high_value = 10**(14)
    max_jerk = very_high_value
    max_juke = very_high_value
    max_velocity = 10
    max_kappa = 0.2
    max_accel = 1
    max_pinch = 3/ELL
    max_deviation_from_reference = very_high_value
    epsilon = 0.01 #tolerance for convergence of the solution
    action = np.zeros(3)

    max_theta = np.pi

    solver = NLPSolver(
        num_time_steps = 100,
        duration = 2,
        initial_position = initial_position,
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
    
    def get_next_state(state, action):
        h = 100/2
        next_state = np.zeros_like(state)
        next_state[0] = state[0] + h * state[3] * np.cos(state[2])  # xpos 
        next_state[1] = state[1] + h * state[3] * np.sin(state[2])  # ypos
        next_state[2] = state[2] + h * state[3] * state[4]          # theta
        next_state[3] = state[3] + h * state[5]                     # velocity
        next_state[4] = state[4] + h * state[6]                     # kappa
        next_state[5] = state[5] + h * action[0]                    # accel
        next_state[6] = state[6] + h * action[1]                    # pinch
        return next_state

    def rollout_actions(state, actions):
        #assert len(actions.shape) == 2 and actions.shape[1] == 3
        #assert len(state.shape) == 1 and state.shape[0] == 7
        num_time_steps = actions.shape[0]
        state_trajectory = np.zeros((num_time_steps+1, state.shape[0]))
        state_trajectory[0] = state
        for k in range(num_time_steps):
            state_trajectory[k+1] = get_next_state(state_trajectory[k], actions[k])
        return state_trajectory
    
    num_actions = 3
    zero_action = np.zeros((solver.num_time_steps, num_actions)) 
    zero_action_state_trajectory = rollout_actions(np.concatenate([initial_position,np.array([0,0,0])]), zero_action)
    #current_state = initial_state
    prev_state_trajectory = zero_action_state_trajectory
    prev_input_trajectory = zero_action   
        
    
    solver.update_state(initial_state, trajectory_init = "rollout", rollout_states = zero_action_state_trajectory, rollout_actions = zero_action)

    n = 7*(solver.num_time_steps+1)+2*(solver.num_time_steps)
    m = 7*(solver.num_time_steps)

    print("the number of variables is....")
    print(n)

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
    #lb[theta_idx:kappa_idx] = -solver.constants["max_theta"]
    lb[theta_idx:kappa_idx] = -np.inf
    lb[kappa_idx:accel_idx] = -solver.constants["max_kappa"]
    lb[accel_idx:pinch_idx] = -solver.constants["max_accel"]
    lb[pinch_idx:jerk_idx] = -solver.constants["max_pinch"]
    lb[jerk_idx:juke_idx] = -solver.constants["max_jerk"]
    lb[juke_idx:] = -solver.constants["max_juke"]

    ub[velocity_idx:theta_idx] = solver.constants["max_velocity"]
    #ub[theta_idx:kappa_idx] = solver.constants["max_theta"]
    ub[theta_idx:kappa_idx] = np.inf
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
    nlp.add_option('max_iter', 1000) 

    #NUM_TIME_STEPS = 100
    NUM_TIME_STEPS = 1
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
            
            increment = 100 + 1
            xpos = x[increment*0:increment*1]
            ypos = x[increment*1:increment*2]
            velocity = x[increment*2:increment*3]
            theta = x[increment*3:increment*4]
            kappa = x[increment*4:increment*5]
            accel = x[increment*5:increment*6]
            pinch = x[increment*6:increment*7]
            increment2 = 100
            jerk = x[increment*7:increment*7+increment2]
            juke = x[increment*7+increment2:]
            ax[0,0].scatter(xpos, ypos, c='black', label = 'Planned trajectory')
            ax[0,0].scatter(solver.current_state.xpos, solver.current_state.ypos, s=30, c='blue')
            #ax[0,0].scatter(solver.constants.final_position.value[0], solver.constants.final_position.value[1], s=30, c='red')
            ax[0,0].scatter(solver.constants.final_position[0], solver.constants.final_position[1], s=30, c='red')
            ax[0,1].plot(np.arange(solver.num_time_steps+1), velocity)
            ax[0,2].plot(np.arange(solver.num_time_steps+1), accel)
            ax[1,0].plot(np.arange(solver.num_time_steps+1), theta)
            ax[1,1].plot(np.arange(solver.num_time_steps+1), kappa)
            ax[1,2].plot(np.arange(solver.num_time_steps+1), np.arctan(ELL * kappa))
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
        action[2] = 0 # brake action - not used for our purposes

        # Step through the environment
        observation, reward, done, info = env.step(action)

        solver.shift_prev_trajectory() # Shift to have proper initialization after one solve.

        if done:
            observation = env.reset()
    env.close
    
    ax[0,0].scatter(actual_trajectory[:,0], actual_trajectory[:,1], c='green', label = "actual trajectory")
    ax[0,1].plot(np.arange(NUM_TIME_STEPS), actual_trajectory[:,2])
    ax[1,0].plot(np.arange(NUM_TIME_STEPS), actual_trajectory[:,3])
    ax[1,1].plot(np.arange(NUM_TIME_STEPS), actual_trajectory[:,4])
    ax[1,2].plot(np.arange(NUM_TIME_STEPS), np.arctan(ELL * actual_trajectory[:,4]))


    plt.show()
    plt.savefig("scp_trajectory.png")