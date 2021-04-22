import cvxpy as cp
import numpy as np
import env as car_env 
import matplotlib.pyplot as plt

from typing import List, Dict, Tuple
from copy import deepcopy
from dataclasses import dataclass

ELL = 80 + 82 # Length of car; defined in neural-car-dynamics global variables

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
            final_position: np.ndarray,
            max_jerk: float,
            max_juke: float,
            max_velocity: float,
            max_kappa: float,
            max_deviation_from_reference: float,
            obstacles: List[Obstacle] = [],
            solver = cp.SCS
        ):
        self.num_time_steps = num_time_steps
        self.duration = duration
        time_step_magnitude = duration / num_time_steps
        self.solver = solver
        # Parameter semantics:
        #   State variables:
        #   - self.variables.xpos[0] is the initial state
        #   Inputs:
        #   - state[t+1] = f(state[t], input[t])

        self.constants = AttrDict.from_dict({
            "time_step_magnitude": cp.Constant(time_step_magnitude),
            "final_position": cp.Constant(final_position),
            "max_jerk": cp.Constant(max_jerk),
            "max_juke": cp.Constant(max_juke),
            "max_velocity": cp.Constant(max_velocity),
            "max_kappa": cp.Constant(max_kappa),
            "max_deviation_from_reference": cp.Constant(max_deviation_from_reference),
        })
        self.obstacles = obstacles

        # Store the trajectories of the previous iterate
        # This has to be feasible!
        self.previous_trajectory = AttrDict.from_dict({ 
            state_variable_name: cp.Parameter(shape = num_time_steps+1) \
                for state_variable_name in self.state_variable_names    
        })
        # Current state 
        self.current_state = AttrDict.from_dict({
            state_variable_name: cp.Parameter() \
                for state_variable_name in self.state_variable_names    
        })
        self.variables = AttrDict.from_dict({
            # Note: Idiomatic way to combine two dictionaries
            **{state_variable_name: cp.Variable(shape = num_time_steps+1) \
                for state_variable_name in self.state_variable_names},
            **{input_variable_name: cp.Variable(shape = num_time_steps) \
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
                value = self.previous_trajectory[key].value[1]
            assert value is not None
            self.current_state[key].value = value       

        if trajectory_init == "zero": 
            self._init_trajectory_zero()
        else:
            raise ValueError(f"Trajectory initializatoin {trajectory_init} not recognized")

    def _init_trajectory_zero(self):
        """ Initialize the previous trajectory to the trajectory defined by zero input for all time 
        
        I.e. car moves with fixed constant velocity 
        """
        xpos = self.current_state.xpos.value
        ypos = self.current_state.ypos.value
        veloc = self.current_state.velocity.value 
        theta = self.current_state.theta.value
        h = self.constants.time_step_magnitude.value
        # TODO: Ask polo to check this
        vx, vy = veloc * np.cos(theta), veloc * np.sin(theta)

        self.previous_trajectory.xpos.value = xpos + vx * h * np.arange(self.num_time_steps+1)
        self.previous_trajectory.ypos.value = ypos + vy * h * np.arange(self.num_time_steps+1)
        self.previous_trajectory.velocity.value = veloc * np.ones(self.num_time_steps+1)
        self.previous_trajectory.theta.value = theta * np.ones(self.num_time_steps+1)
        self.previous_trajectory.kappa.value = np.zeros(self.num_time_steps+1)
        self.previous_trajectory.accel.value = np.zeros(self.num_time_steps+1)
        self.previous_trajectory.pinch.value = np.zeros(self.num_time_steps+1)
        
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
        input_norm_sq = cp.norm(input, axis=0)**2
        assert input_norm_sq.shape == (self.num_time_steps,)
        return cp.Minimize(
            cp.sum(input_norm_sq) \
            + cp.norm(self.position[:,-1] - self.constants.final_position, p=1)
        )

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

        prev_xpos = self.previous_trajectory.xpos
        prev_ypos = self.previous_trajectory.ypos
        prev_veloc = self.previous_trajectory.velocity
        prev_theta = self.previous_trajectory.theta
        prev_kappa = self.previous_trajectory.kappa
        prev_accel = self.previous_trajectory.accel
        prev_pinch = self.previous_trajectory.pinch

        h = self.constants.time_step_magnitude
        r = self.constants.max_deviation_from_reference
        delta_theta = curr(theta) - curr(prev_theta) # 0i - 0i^{(k)}

        constraints = []

        """ Add the dynamics constraints """
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
        ]
        """ Add the current state constraints """
        constraints += [
            xpos[0] == self.current_state.xpos,
            ypos[0] == self.current_state.ypos,
            veloc[0] == self.current_state.velocity,
            theta[0] == self.current_state.theta, 
            kappa[0] == self.current_state.kappa, 
            accel[0] == self.current_state.accel, 
            pinch[0] == self.current_state.pinch
        ]

        """ Add the control constraints """
        constraints += [
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

        # TODO: Add the max deviation from reference constraint
        constraints += [

        ]

        return constraints

    def _convex_solve(self) -> Tuple[float, float]:
        """ 
        Perform one convex solve as part of SCP

        :return cost: The cost of the current solution         
        :return diff: The difference in the norm of the input trajectories
            diff = None if this is the first iteration
        """
        diff = None
        # Store the previous inputs for comparison
        prev_input = self.input.value

        optval = self.problem.solve(solver = self.solver)

        if self.problem.status in ["infeasible", "unbounded"]:
            raise Exception(f"The problem was {self.problem.status}")
        
        if prev_input is not None:
            curr_input = self.input.value
            diff = np.linalg.norm(prev_input - curr_input)

        # Update the previous trajectory
        for state_variable_name in self.state_variable_names:
            self.previous_trajectory[state_variable_name].value = self.variables[state_variable_name].value
        return optval, diff

    def solve(self, tol = 1e-7, max_iters: int = np.inf, verbose = False) -> float:
        """ Perform sequential convex solves to find a locally optimal solution
        """
        self.problem = cp.Problem(self.objective, self.constraints)
        num_iters = 0
        diff = tol + 1 

        print("Starting a new SCP solve")
        while diff is None or diff > tol:
            cost, diff = self._convex_solve()
            if num_iters >= max_iters:
                break 
            num_iters += 1
            if verbose:
                print(self.problem.status, cost, diff)
        return cost
            
def rotate_by_angle(vec, th):
    M = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
    return M@vec

def get_current_state(env) -> Dict[str, float]:
    x = (1/2)*(env.car.wheels[2].position[0]+env.car.wheels[3].position[0])
    y = (1/2)*(env.car.wheels[2].position[1]+env.car.wheels[3].position[1])
    theta = env.car.hull.angle + np.pi / 2
    vec1 = np.array(env.car.hull.linearVelocity) # Velocity as a vector
    vec2 = rotate_by_angle(np.array([1,0]), theta)
    dot_prod = np.dot(vec1, vec2)
    velocity = np.linalg.norm(vec1,2) if dot_prod > 0 else -np.linalg.norm(vec1,2)
    kappa = np.tan(env.car.wheels[0].angle) / ELL

    return {
        "xpos": x,
        "ypos": y,
        "velocity": velocity,
        "theta": theta,
        "kappa": kappa,
        "accel": None, 
        "pinch": None
    }

def plot_trajectory(solver):
    plt.scatter(solver.variables.xpos.value, solver.variables.ypos.value, c='black', label = 'Planned trajectory')
    plt.scatter(solver.current_state.xpos.value, solver.current_state.ypos.value, s=10, c='blue')
    plt.scatter(solver.constants.final_position.value[0], solver.constants.final_position.value[1], s=10, c='red')

def main():
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
    # Right now the get_current_state returns None for these
    initial_state['accel'] = 0
    initial_state['pinch'] = 0

    x, y = initial_state['xpos'], initial_state['ypos']
    theta = initial_state['theta']
    direction = np.array([np.cos(theta), np.sin(theta)])
    final_position = np.array([x,y]) + 100 * direction
    
    # Initialize to very high value until further notice
    very_high_value = 10**(14)
    max_jerk = 100000
    max_juke = 100000
    max_velocity = 10
    max_kappa = 0.2
    max_deviation_from_reference = very_high_value
    epsilon = 0.01 #tolerance for convergence of the solution
    action = np.zeros(3)

    solver = SCPSolver(
        num_time_steps = 100, 
        duration = 2, 
        final_position = final_position,
        max_jerk = max_jerk, 
        max_juke = max_juke, 
        max_velocity = max_velocity,
        max_kappa = max_kappa, 
        max_deviation_from_reference = max_deviation_from_reference,
        solver = cp.SCS
    )
    solver.update_state(initial_state)


    NUM_TIME_STEPS = 50
    actual_trajectory = np.zeros([NUM_TIME_STEPS, 7])
    first = True
    fig, ax = plt.subplots(2,3)

    for _ in range(NUM_TIME_STEPS):
        env.render()
        cost: float = solver.solve(tol = epsilon, max_iters=1000, verbose=True)
        
        if first: 
            ax[0,0].scatter(solver.variables.xpos.value, solver.variables.ypos.value, c='black', label = 'Planned trajectory')
            ax[0,0].scatter(solver.current_state.xpos.value, solver.current_state.ypos.value, s=10, c='blue')
            ax[0,0].scatter(solver.constants.final_position.value[0], solver.constants.final_position.value[1], s=10, c='red')
            ax[0,1].plot(np.arange(solver.num_time_steps+1), solver.variables.velocity.value)
            ax[1,0].plot(np.arange(solver.num_time_steps+1), solver.variables.theta.value)
            ax[1,1].plot(np.arange(solver.num_time_steps+1), solver.variables.kappa.value)
            ax[1,2].plot(np.arange(solver.num_time_steps+1), np.arctan(ELL * solver.variables.kappa.value))
            first = False 

        # Obtain the chosen action given the MPC solve
        kappa = solver.variables.kappa[0].value
        action[0] = np.arctan(ELL * kappa) # steering action
        SIZE = 0.02
        mass = 1000000*SIZE*SIZE # friction ~= mass (as stated in dynamics)
        acc = solver.variables.accel[0].value
        action[1] = mass*acc # gas action
        action[2] = 0 # brake action - not used for our purposes
        
        # Step through the environment
        observation, reward, done, info = env.step(action)
        if done:
            observation = env.reset()

        # Update the solver state
        state: Dict[str, float] = get_current_state(env)
        solver.update_state(state)
        actual_trajectory[_] = np.array([
            state['xpos'], 
            state['ypos'], 
            state['velocity'], 
            state['theta'],
            state['kappa'],
            state['accel'],
            state['pinch']
        ])
    env.close()

    ax[0,0].scatter(actual_trajectory[:,0], actual_trajectory[:,1], c='green', label = "actual trajectory")
    ax[0,1].plot(np.arange(NUM_TIME_STEPS), actual_trajectory[:,2])
    ax[1,0].plot(np.arange(NUM_TIME_STEPS), actual_trajectory[:,3])
    ax[1,1].plot(np.arange(NUM_TIME_STEPS), actual_trajectory[:,4])
    ax[1,2].plot(np.arange(NUM_TIME_STEPS), np.arctan(ELL * actual_trajectory[:,4]))

    plt.show()
    plt.savefig("scp_trajectory.png")

if __name__ == "__main__":
    main()