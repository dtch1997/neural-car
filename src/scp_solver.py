import cvxpy as cp
import numpy as np
import env as car_env
import matplotlib.pyplot as plt

from typing import List, Dict, Tuple
from copy import deepcopy
from dataclasses import dataclass

SIZE = 0.02
ELL = SIZE*(80 + 82) # Length of car; defined in neural-car-dynamics global variables

""" Slicing convenience functions """
def nxt(var: cp.Variable):
    return var[1:]
def curr(var: cp.Variable):
    return var[:-1]

# @dataclass
# class Obstacle:
#     xpos: float
#     ypos: float
#     radius: float

#     @property
#     def position(self) -> np.ndarray:
#         return np.array([self.xpos, self.ypos])

# class AttrDict(Dict):
#     """ Dictionary that also lets you get the entries as properties """
#     def __setitem__(self, key, value):
#         super().__setitem__(key, value)
#         setattr(self, key, value)
#     @staticmethod
#     def from_dict(dict):
#         attrdict =  AttrDict()
#         for key, value in dict.items():
#             attrdict[key] = value # Calls __setitem_
#         return attrdict

# class SCPSolver:
#     """ A sequential convex programming solver for the CarRacing OpenAI gym environment

#     Usage:

#     solver = SCPSolver(...)
#     for time_step in range(MAX_TIME_STEPS):

#     """

#     state_variable_names = ["xpos", "ypos", "velocity", "theta", "kappa", "accel", "pinch"]
#     input_variable_names = ["jerk", "juke"]

#     def __init__(self,
#             num_time_steps: float,
#             duration: float,
#             final_position: np.ndarray,
#             max_jerk: float,
#             max_juke: float,
#             max_velocity: float,
#             max_kappa: float,
#             max_accel: float,
#             max_pinch: float,
#             max_theta: float,
#             max_deviation_from_reference: float,
#             obstacles: List[Obstacle] = [],
#             solver = cp.SCS
#         ):
#         self.num_time_steps = num_time_steps
#         self.duration = duration
#         time_step_magnitude = duration / num_time_steps
#         self.solver = solver
#         # Parameter semantics:
#         #   State variables:
#         #   - self.variables.xpos[0] is the initial state
#         #   Inputs:
#         #   - state[t+1] = f(state[t], input[t])
#         """
#         self.constants = AttrDict.from_dict({
#             "time_step_magnitude": cp.Constant(time_step_magnitude),
#             "final_position": cp.Constant(final_position),
#             "max_jerk": cp.Constant(max_jerk),
#             "max_juke": cp.Constant(max_juke),
#             "max_velocity": cp.Constant(max_velocity),
#             "max_kappa": cp.Constant(max_kappa),
#             "max_accel": cp.Constant(max_accel),
#             "max_pinch": cp.Constant(max_pinch),
#             "max_theta": cp.Constant(max_theta),
#             "max_deviation_from_reference": cp.Constant(max_deviation_from_reference),
#         })
#         """
#         self.constants = AttrDict.from_dict({
#             "time_step_magnitude": time_step_magnitude,
#             "final_position": final_position,
#             "max_jerk": max_jerk,
#             "max_juke": max_juke,
#             "max_velocity": max_velocity,
#             "max_kappa": max_kappa,
#             "max_accel": max_accel,
#             "max_pinch": max_pinch,
#             "max_theta": max_theta,
#             "max_deviation_from_reference": max_deviation_from_reference,
#         })
#         self.obstacles = obstacles

#         # Store the trajectories of the previous iterate
#         # This has to be feasible!
#         self.previous_trajectory = AttrDict.from_dict({
#             state_variable_name: cp.Parameter(shape = num_time_steps+1) \
#                 for state_variable_name in self.state_variable_names
#         })
#         # Current state
#         self.current_state = AttrDict.from_dict({
#             state_variable_name: cp.Parameter() \
#                 for state_variable_name in self.state_variable_names
#         })
#         self.variables = AttrDict.from_dict({
#             # Note: Idiomatic way to combine two dictionaries
#             **{state_variable_name: cp.Variable(shape = num_time_steps+1) \
#                 for state_variable_name in self.state_variable_names},
#             **{input_variable_name: cp.Variable(shape = num_time_steps) \
#                 for input_variable_name in self.input_variable_names}
#         })

#     def update_state(self, values: Dict[str, float], trajectory_init = "zero"):
#         """ Update the current state of the car in the solver

#         Also initializes a feasible trajectory from that state
#         By default, this is the trajectory obtained by having zero input

#         Usage:
#             solver.update_state({
#                 "xpos": 23.4,
#                 "ypos": 14.5,
#                 ...
#                 "accel": 0.1,
#                 "pinch: 1.2
#             })
#         """
#         for key, value in values.items():
#             assert key in self.state_variable_names, "Invalid state variable entered"
#             if value is None:
#                 value = self.previous_trajectory[key].value[1]
#             assert value is not None
#             self.current_state[key].value = value

#         if trajectory_init == "zero":
#             self._init_trajectory_zero()
#         else:
#             raise ValueError(f"Trajectory initializatoin {trajectory_init} not recognized")

#     def _init_trajectory_zero(self):
#         """ Initialize the previous trajectory to the trajectory defined by zero input for all time

#         I.e. car moves with fixed constant velocity
#         """
#         xpos = self.current_state.xpos.value
#         ypos = self.current_state.ypos.value
#         veloc = self.current_state.velocity.value
#         theta = self.current_state.theta.value
#         #h = self.constants.time_step_magnitude.value
#         h = self.constants.time_step_magnitude
#         # TODO: Ask polo to check this
#         vx, vy = veloc * np.cos(theta), veloc * np.sin(theta)

#         self.previous_trajectory.xpos.value = xpos + vx * h * np.arange(self.num_time_steps+1)
#         self.previous_trajectory.ypos.value = ypos + vy * h * np.arange(self.num_time_steps+1)
#         self.previous_trajectory.velocity.value = veloc * np.ones(self.num_time_steps+1)
#         self.previous_trajectory.theta.value = theta * np.ones(self.num_time_steps+1)
#         self.previous_trajectory.kappa.value = np.zeros(self.num_time_steps+1)
#         self.previous_trajectory.accel.value = np.zeros(self.num_time_steps+1)
#         self.previous_trajectory.pinch.value = np.zeros(self.num_time_steps+1)

#     @property
#     def input(self):
#         """ Get all the variables that encode the input to the system """
#         return cp.vstack([
#             self.variables.jerk,
#             self.variables.juke
#         ])

#     @property
#     def position(self):
#         return cp.vstack([
#             self.variables.xpos,
#             self.variables.ypos,
#             self.variables.velocity,
#             self.variables.theta
#         ])

#     @property
#     def state(self):
#         """ Get all the variables that encode the state of the system """
#         return cp.vstack([
#             self.variables.xpos,
#             self.variables.ypos,
#             self.variables.velocity,
#             self.variables.theta,
#             self.variables.kappa,
#             self.variables.accel,
#             self.variables.pinch
#         ])

#     @property
#     def objective(self):
#         input = cp.vstack([self.variables.jerk, self.variables.juke])
#         assert input.shape == (2, self.num_time_steps)
#         input_norm_sq = cp.norm(input, axis=0)**2
#         assert input_norm_sq.shape == (self.num_time_steps,)
#         return cp.Minimize(
#             cp.sum(input_norm_sq) \
#             + cp.norm(self.position[:,-1] - self.constants.final_position, p=1)
#         )

#     @property
#     def constraints(self):
#         xpos = self.variables.xpos
#         ypos = self.variables.ypos
#         veloc = self.variables.velocity
#         theta = self.variables.theta
#         kappa = self.variables.kappa
#         accel = self.variables.accel
#         pinch = self.variables.pinch
#         jerk = self.variables.jerk
#         juke = self.variables.juke

#         prev_xpos = self.previous_trajectory.xpos
#         prev_ypos = self.previous_trajectory.ypos
#         prev_veloc = self.previous_trajectory.velocity
#         prev_theta = self.previous_trajectory.theta
#         prev_kappa = self.previous_trajectory.kappa
#         prev_accel = self.previous_trajectory.accel
#         prev_pinch = self.previous_trajectory.pinch

#         h = self.constants.time_step_magnitude
#         r = self.constants.max_deviation_from_reference
#         delta_theta = curr(theta) - curr(prev_theta) # 0i - 0i^{(k)}

#         constraints = []

#         """ Add the dynamics constraints """
#         """
#         constraints += [
#             nxt(xpos) == curr(xpos) + h * (
#                     cp.multiply(curr(veloc), cp.cos(curr(prev_theta)))
#                     - cp.multiply(cp.multiply(curr(prev_veloc), np.sin(curr(prev_theta.value))), delta_theta)
#                 ),
#             nxt(ypos) == curr(ypos) + h * (
#                     cp.multiply(curr(veloc), np.sin(curr(prev_theta.value)))
#                     + cp.multiply(cp.multiply(curr(prev_veloc), np.cos(curr(prev_theta.value))), delta_theta)
#                 ),
#             nxt(theta) == curr(theta) + h * (
#                     cp.multiply(curr(prev_veloc), curr(prev_kappa))
#                     + cp.multiply(curr(prev_kappa), curr(veloc) - curr(prev_veloc))
#                     + cp.multiply(curr(prev_veloc), curr(kappa) - curr(prev_kappa))
#                 ),
#             nxt(veloc) == curr(veloc) + h * curr(accel),
#             nxt(kappa) == curr(kappa) + h * curr(pinch),
#             nxt(accel) == curr(accel) + h * jerk,
#             nxt(pinch) == curr(pinch) + h * juke,
#         ]
#         """
#         constraints += [
#             nxt(xpos) == curr(xpos) + h * (
#                     cp.multiply(curr(veloc), np.cos(curr(prev_theta.value)))
#                     - cp.multiply(np.multiply(curr(prev_veloc.value), np.sin(curr(prev_theta.value))), delta_theta)
#                 ),
#             nxt(ypos) == curr(ypos) + h * (
#                     cp.multiply(curr(veloc), np.sin(curr(prev_theta.value)))
#                     + cp.multiply(cp.multiply(curr(prev_veloc), np.cos(curr(prev_theta.value))), delta_theta)
#                 ),
#             nxt(theta) == curr(theta) + h * (
#                     cp.multiply(curr(prev_veloc), curr(prev_kappa))
#                     + cp.multiply(curr(prev_kappa), curr(veloc) - curr(prev_veloc))
#                     + cp.multiply(curr(prev_veloc), curr(kappa) - curr(prev_kappa))
#                 ),
#             nxt(veloc) == curr(veloc) + h * curr(accel),
#             nxt(kappa) == curr(kappa) + h * curr(pinch),
#             nxt(accel) == curr(accel) + h * jerk,
#             nxt(pinch) == curr(pinch) + h * juke,
#         ]
#         """ Add the current state constraints """
#         constraints += [
#             xpos[0] == self.current_state.xpos,
#             ypos[0] == self.current_state.ypos,
#             veloc[0] == self.current_state.velocity,
#             theta[0] == self.current_state.theta,
#             kappa[0] == self.current_state.kappa,
#             accel[0] == self.current_state.accel,
#             pinch[0] == self.current_state.pinch
#         ]

#         """ Add the control constraints """
#         constraints += [
#             #veloc[-1] == 0,
#             #xpos[-1] == self.constants.final_position[0],
#             #ypos[-1] == self.constants.final_position[1],
#             cp.norm(veloc, p=np.inf) <= self.constants.max_velocity,
#             #cp.norm(kappa, p=np.inf) <= self.constants.max_kappa,
#             kappa <= self.constants.max_kappa,
#             kappa >= -self.constants.max_kappa,
#             cp.norm(accel, p=np.inf) <= self.constants.max_accel,
#             cp.norm(jerk, p=np.inf) <= self.constants.max_jerk,
#             cp.norm(juke, p=np.inf) <= self.constants.max_juke,
#             cp.norm(pinch, p=np.inf) <= self.constants.max_pinch,
#             theta <= self.constants.max_theta,
#             theta >= -self.constants.max_theta
#         ]

#         # TODO: Add the obstacle avoidance constraints
#         constraints += [

#         ]

#         # TODO: Add the max deviation from reference constraint
#         constraints += [

#         ]

#         return constraints

#     def _convex_solve(self) -> Tuple[float, float]:
#         """
#         Perform one convex solve as part of SCP

#         :return cost: The cost of the current solution
#         :return diff: The difference in the norm of the input trajectories
#             diff = None if this is the first iteration
#         """
#         diff = None
#         # Store the previous inputs for comparison
#         prev_input = self.input.value

#         optval = self.problem.solve(solver = self.solver)

#         if self.problem.status in ["infeasible", "unbounded"]:
#             raise Exception(f"The problem was {self.problem.status}")

#         if prev_input is not None:
#             curr_input = self.input.value
#             diff = np.linalg.norm(prev_input - curr_input)

#         # Update the previous trajectory
#         for state_variable_name in self.state_variable_names:
#             self.previous_trajectory[state_variable_name].value = self.variables[state_variable_name].value
#         return optval, diff

#     def solve(self, tol = 1e-7, max_iters: int = np.inf, verbose = False) -> float:
#         """ Perform sequential convex solves to find a locally optimal solution
#         """
#         self.problem = cp.Problem(self.objective, self.constraints)
#         num_iters = 0
#         diff = tol + 1

#         print("Starting a new SCP solve")
#         while diff is None or diff > tol:
#             cost, diff = self._convex_solve()
#             if num_iters >= max_iters:
#                 break
#             num_iters += 1
#             if verbose:
#                 print(self.problem.status, cost, diff)
#         return cost

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
    kappa_mpc = np.tan(env.car.wheels[0].joint.angle) / ELL

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
    initial_state['accel'] = 0
    initial_state['pinch'] = 0

    x_extract, y = initial_state['xpos'], initial_state['ypos']
    theta = initial_state['theta']
    direction = np.array([np.cos(theta), np.sin(theta),0,0])
    orth_direction = np.array([*rotate_by_angle(direction[:2], np.pi/2),0,0])
    final_position = np.array([x_extract,y,0,theta]) + 10 * direction + 10 * orth_direction

    print("Initial x: ",x_extract)
    print("Initial y: ", y)
    print("Final x: ", final_position[0])
    print("Final y: ", final_position[1])

    # # Initialize to very high value until further notice
    # very_high_value = 10**(14)
    # max_jerk = very_high_value #100000
    # max_juke = very_high_value #100000
    # max_velocity = 10
    # max_kappa = np.tan(0.4200316) / ELL

    # max_accel = 1
    # max_pinch = 3/ELL
    # max_deviation_from_reference = very_high_value
    # epsilon = 0.01 #tolerance for convergence of the solution
    action = np.zeros(3)
    # max_theta = very_high_value #np.pi #no need for a maximum theta value

    # solver = SCPSolver(
    #     num_time_steps = 600,
    #     duration = 2,
    #     final_position = final_position,
    #     max_jerk = max_jerk,
    #     max_juke = max_juke,
    #     max_velocity = max_velocity,
    #     max_kappa = max_kappa,
    #     max_accel = max_accel,
    #     max_pinch = max_pinch,
    #     max_theta = max_theta,
    #     max_deviation_from_reference = max_deviation_from_reference,
    #     solver = cp.SCS
    # )
    # solver.update_state(initial_state)


    NUM_TIME_STEPS = 1200
    actual_trajectory = np.zeros([NUM_TIME_STEPS, 7])
    first = True
    fig, ax = plt.subplots(2,3)

    # derivative = np.zeros(100)
    # prev_velocity = 0

    ##PORTED CODE STARTS HERE (MORE OR LESS)
	#DEFINITION OF CONSTANTS FOR NAVIGATION
    NavigatioN = 800 #number of time steps in a single solve
    #xFinal = np.array([x_extract,y,theta,0,0,0,0]) + np.array([10,10,0,0,0,0,0]) * np.hstack((direction,np.array([0,0,0]))) + np.array([10,10,0,0,0,0,0]) * np.hstack((orth_direction,np.array([0,0,0]))) #coordinates of goal location and characteristics
    #xFinal = np.array([x_extract,y,theta,0,0,0,0]) + np.array([-10,10,0,0,0,0,0])
    xFinal = np.array([x_extract,y,theta,0,0,0,0]) + 30 * np.hstack((direction,np.array([0,0,0]))) + 8 * np.hstack((orth_direction,np.array([0,0,0])))
    print("target: ",xFinal)

    h = 0.02 #50Hz
    # h = 0.04 #25Hz (debug)
    epsilon = 2e-2 #to check convergence of optimizer
    state_epsilon = 2e-2 #to check drift from controller trajectory

    #DECLARATION OF CONTROL LIMITS
    speed_limit = 20
    kappa_limit = np.tan(0.4) / ELL
    accel_limit = 1
    pinch_limit = 3/ELL

    #DECLARATION OF POSITIONS OF PERMANENT OBSTACLES FOR OBSTACLE AVOIDANCE
    zObs = np.array([[5,2],[12,6],[12,1],[20,3],[26,6],[26,1]]) #positions of center of obstacles
    # zObs = np.array([[5,2]])
    zObs = np.apply_along_axis(lambda equis: rotate_by_angle(equis,theta), 1, zObs)
    zObs = np.array([x_extract,y]) + zObs
    rObs = np.array([[2.06],[1.78],[1.78],[3.81],[1.66],[1.56]]) #radii of obstacles
    # rObs = np.array([[2.06]])
    obstacles = np.shape(rObs)[0] #total number of obstacles to dodge
    rTotal = np.repeat(rObs,NavigatioN,axis=1) #repeated instances of the obstacle radii, one for each time step (for obstacle avoidance)

    #DECLARATION OF TRAJECTORY VALUES
    xt = np.random.rand(NavigatioN,7) #arbitrary stand-in for previous solution state trajectory
    ut = np.random.rand(NavigatioN,2) #arbitrary stand-in for previous solution control trajectory
    # zt = np.random.rand(NavigatioN,2) #arbitrary stand-in for trajectory values pertaining specifically to x,y positions (for obstacle avoidance)

    for _ in range(NUM_TIME_STEPS):
        env.render()

        #cost: float = solver.solve(tol = epsilon, max_iters=1000, verbose=True)

        if first:
			#DEFINITION OF INITIAL TRAJECTORY ROLLOUT AT FIRST STEP
            xInitial = np.array([initial_state['xpos'],initial_state['ypos'],initial_state['theta'],0,0,0,0]) #Coordinates of initial position of car: x,y,theta,V,kappa,acceleration,pinch
            #print("start: ",xInitial)
            xt = np.repeat(np.reshape(xInitial,(1,np.shape(xInitial)[0])),NavigatioN,axis=0)
            # for i in range(1,NavigatioN):
            #     # xt[i,:] = xInitial+(((xFinal-xInitial)/NavigatioN)*i) #linear interpolation for lowest cost path with no obstacles and no dynamics constraints
            #     xt[i,:] = xInitial #rollout from no control inputs, from static start
            state_diff = np.inf

        else:
            #INITIAL TRAJECTORY ROLLOUT FOR STEPS BEYOND THE FIRST
            initial_state = get_current_state(env) #update current state values from environment
            initial_state['accel'] = xt[1,5] #carried over from previous solution
            initial_state['pinch'] = xt[1,6] #carried over from previous solution
            xInitial = np.array([initial_state['xpos'],initial_state['ypos'],initial_state['theta'],initial_state['velocity'],initial_state['kappa'],initial_state['accel'],initial_state['pinch']]) #current state defines starting point for solve
            xt[:-1,:] = xt[1:,:] #use of previous solution as initialization for trajectory (last state is kept from last time, a duplicate)
            ut[:-1,:] = ut[1:,:]
            state_diff = np.linalg.norm(xt[0,:] - xInitial,ord=2) #discrepancy between solver estimation of next state and recovered state
            print('error in state estimation: ',state_diff)
            #print(ut)

        #INITIALIZATION OF LOOP CONTROL VALUES
        diff = np.inf #initialize to unreasonable value to overwrite in loop

        while abs(diff) > epsilon and state_diff > state_epsilon:
            # #CALCULATION OF OBJECTIVE VALUE WITH PRIOR SOLUTION
            # prevGas = h*np.linalg.norm(ut,'fro')**2 + np.linalg.norm(xFinal - xt[-1,:],1) #objective value with prior trajectory
            # print('prev: ',prevGas)

            prev_theta = xt[:,2] #history of angles
            prev_veloc = xt[:,3] #history of velocities
            prev_kappa = xt[:,4] #history of curvatures

            #SOLVING TRAJECTORY AS CONVEX PROBLEM
            x = cp.Variable((NavigatioN,7)) #state trajectory
            u = cp.Variable((NavigatioN,2)) #control inputs
            #velVec = cp.Variable((NavigatioN,2)) #vector of instantaneous acceleration (for traction control)

            objectNav = cp.Minimize(h*cp.square(cp.norm(u,'fro')) + cp.norm(xFinal - x[-1,:],1)) #objective function
            
            constrNav = [] #initialize constraints list

            #INITIAL POSITION
            constrNav += [x[0,:] == xInitial] #hard constraint on initial position
            #constrNav += [x[NavigatioN-1,:] == xFinal]

            #DYNAMICS CONSTRAINTS
            constrNav += [
                    nxt(x[:,0]) == curr(x[:,0]) + h * ( #xpos constraint x[:,0]
                            cp.multiply(curr(x[:,3]), np.cos(curr(prev_theta)))
                            - cp.multiply(cp.multiply(curr(prev_veloc), np.sin(curr(prev_theta))), curr(x[:,2] - prev_theta))
                        ),
                    nxt(x[:,1]) == curr(x[:,1]) + h * ( #ypos constraint x[:,1]
                            cp.multiply(curr(x[:,3]), np.sin(curr(prev_theta)))
                            + cp.multiply(cp.multiply(curr(prev_veloc), np.cos(curr(prev_theta))), curr(x[:,2] - prev_theta))
                        ),
                    nxt(x[:,2]) == curr(x[:,2]) + h * ( #theta constraint x[:,2]
                            cp.multiply(curr(x[:,3]), curr(prev_kappa))
                            + cp.multiply(curr(prev_veloc), curr(x[:,4]) - curr(prev_kappa))
                        ),
                    nxt(x[:,3]) == curr(x[:,3]) + h * curr(x[:,5]), #velocity constraint x[:,3]
                    nxt(x[:,4]) == curr(x[:,4]) + h * curr(x[:,6]), #kappa constraint x[:,4]
                    nxt(x[:,5]) == curr(x[:,5]) + h * curr(u[:,0]), #acceleration constraint x[:,5]
                    nxt(x[:,6]) == curr(x[:,6]) + h * curr(u[:,1]), #pinch constraint x[:,6]
                ]

            #CONTROL LIMIT CONSTRAINTS:
            constrNav += [
                cp.norm(x[1:,3], p=np.inf) <= speed_limit, #max forwards velocity (speed limit)
                #2*cp.multiply(cp.multiply(prev_veloc,prev_kappa),x[:,3] - prev_veloc) + cp.multiply(cp.multiply(prev_veloc,prev_veloc),x[:,4]) <= accel_limit, #max forwards velocity (to prevent slipping)
                x[1:,4] <= kappa_limit, #maximum curvature
                x[1:,4] >= -1*kappa_limit,
                cp.norm(x[:,5], p=np.inf) <= accel_limit, #max acceleration
                cp.norm(x[:,6], p=np.inf) <= pinch_limit #max pinch
                ]

            # #TRACTION CONTROL CONSTRAINTS
            # constrNav += [velVec[:,0] == x[:,5]] #longitudinal acceleration
            # constrNav += [velVec[:,1] == cp.multiply(2*prev_veloc*prev_kappa,x[:,3] - prev_veloc) + cp.multiply(prev_veloc**2,x[:,4])] #centripetal acceleration
            # for i in range(1,NavigatioN): #apply constraint to mutable steps
            #     constrNav += [cp.norm(velVec[i,:],2) <= accel_limit] #skid prevention at step i

            #OBSTACLE AVOIDANCE CONSTRAINTS
            zt = xt[:,:2] #saving only the positional values in the trajectory (for obstacle avoidance)
            zDiffs = cp.Parameter((NavigatioN,2))
            zSub = np.transpose(np.repeat(np.expand_dims(zt,axis=2),obstacles,axis=2),axes=(2,0,1)) \
                - np.transpose(np.repeat(np.expand_dims(zObs,axis=2),NavigatioN,axis=2),axes=(0,2,1)) #OxNx2 tensor containing differences between each position in the trajectory and obstacle center
            zSubNorms = np.linalg.norm(zSub,ord=2,axis=(-1))#OxN matrix of Euclidean distances between each position in the trajectory and each obstacle center
            zDiffs = x[:,:2] - zt #difference between current and prior position trajectory
            
            # zInter = np.einsum('ijk,ij->ik',np.transpose(zSub,axes=(1,2,0)),zDiffs).T #OxN matrix
            # zInter = np.einsum('ijk,ij->ik',np.transpose(zSub,axes=(1,2,0)),x[:,:2] - zt).T #OxN matrix
            # constrNav += [rTotal - zSubNorms - zInter/zSubNorms <= 0] #the truly parallelized version
            
            for o in range(obstacles): #because CVXPY can't use 3D tensors for whatever reason
                constrNav += [rTotal[o,:] - zSubNorms[o,:] - cp.sum(cp.multiply(zSub[o,:,:],zDiffs),axis=1)/zSubNorms[o,:] <= 0] #one constraint per obstacle
            
            # for o in range(0,obstacles): #constraints for each obstacle
            #  	for i in range(1,NavigatioN): #apply constraints to mutable steps individually
            #          constrNav += [rObs[o] - cp.norm((zt[i,:] - zObs[o,:])) - ((zt[i,:] - zObs[o,:])/cp.norm((zt[i,:] - zObs[o,:]))) @ (x[i,:2]-zt[i,:]) <= 0]

            problem = cp.Problem(objectNav,constrNav)
            optval = problem.solve(solver=cp.ECOS)
            #END OF CONVEX OPTIMIZATION PART

            print('opt :',optval) #monitor
            print('problem status: ',problem.status)

            if problem.status == 'infeasible': #if bad information from the environment provokes an infeasible solve, don't crash
                break

            diff = np.max(np.abs(u.value - ut)) #if checking convergence independently of cost function
            xt = deepcopy(x.value) #for use in following iteration
            zt = x[:,:2] #ACTIVATE FOR OBSTACLE AVOIDANCE CONSTRAINTS
            ut = deepcopy(u.value) #for use in following iteration
            # diff = optval - prevGas #for checking convergence via while loop (convergence of cost value)
            print('convergence measure: ',diff)



        if first:
            ax[0,0].scatter(xt[:,0], xt[:,1], c='black', label = 'Planned trajectory') #vehicle trajectory
            #ax[0,0].scatter(xt[:,0], xt[:,1], s=30, c='blue')
            #ax[0,0].scatter(solver.constants.final_position.value[0], solver.constants.final_position.value[1], s=30, c='red')
            ax[0,0].scatter(xFinal[0], xFinal[1], s=30, c='red')
            for o in range(obstacles):
                ax[0,0].scatter(zObs[o,0],zObs[o,1],c='blue')
            ax[0,1].plot(np.arange(NavigatioN), xt[:,3]) #velocity history
            ax[0,2].plot(np.arange(NavigatioN), xt[:,5]) #acceleration history
            ax[1,0].plot(np.arange(NavigatioN), xt[:,2]) #vehicle attitude
            ax[1,1].plot(np.arange(NavigatioN), xt[:,4]) #curvature history
            ax[1,2].plot(np.arange(NavigatioN), np.arctan(ELL * xt[:,4])) #steering angle history

            # print("theta below")
            # print(solver.variables.theta.value)

            first = False

        #print(objectNav)
        print('final opt val: ',optval)

        # Obtain the chosen action given the MPC solve
        kappa = xt[1,4] #copy initial action from last saved trajectory
        action[0] = -1*np.arctan(ELL * kappa) #steering action, rescale
        # mass = 1000000*SIZE*SIZE # friction ~= mass (as stated in dynamics)
        # alpha = (1/43.77365112) # Magicccccc!
        alpha = (1/500) #Magiaaaaaa!
        acc = xt[1,5] #first acceleration value in saved trajector
        action[1] = alpha*acc
        action[2] = 0 # brake action - not used for our purposes

        """
        action[0] = 0
        action[1] = 0.1
        action[2] = 0
        """

        # Step through the environment
        observation, reward, done, info = env.step(action)
        if done:
            observation = env.reset()

        # Update the solver state
        state: Dict[str, float] = get_current_state(env)
        #solver.update_state(state)

        # Plot the trajectory in the MPC coordinates
        actual_trajectory[_] = np.array([
            state['xpos'],
            state['ypos'],
            state['velocity'],
            state['theta'],
            state['kappa'],
            state['accel'],
            state['pinch']
        ])

        #print("Current x: ", state['xpos'])
        #print("Current y: ", state['ypos'])
        #print("Current vel: ", env.car.hull.linearVelocity)
        #print("Current angle:", env.car.hull.angle + np.pi / 2)
        #print(np.arctan(ELL*state['kappa']))
        """
        derivative[_] = (state['velocity'] - prev_velocity)/solver.constants.time_step_magnitude.value
        prev_velocity = state['velocity']
        """


    env.close()
    #print(derivative)

    ax[0,0].scatter(actual_trajectory[:,0], actual_trajectory[:,1], c='green', label = "actual trajectory")
    ax[0,1].plot(np.arange(NUM_TIME_STEPS), actual_trajectory[:,2])
    ax[1,0].plot(np.arange(NUM_TIME_STEPS), actual_trajectory[:,3])
    ax[1,1].plot(np.arange(NUM_TIME_STEPS), actual_trajectory[:,4])
    ax[1,2].plot(np.arange(NUM_TIME_STEPS), np.arctan(ELL * actual_trajectory[:,4]))


    plt.show()
    plt.savefig("scp_trajectory.png")

if __name__ == "__main__":
    main()
