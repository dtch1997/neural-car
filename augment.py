"""
Starter code for the problem "Cart-pole swing-up".

Author: Spencer M. Richards
        Autonomous Systems Lab (ASL), Stanford
        (GitHub: spenrich)
"""

import numpy as np
import h5py

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time

from scipy.integrate import odeint
from src.envs.car import Environment as CarEnv
from dataclasses import dataclass 

@jax.partial(jax.jit, static_argnums=(0,))
def linearize(f, s, u):
    """Linearize the function `f(s,u)` around `(s,u)`."""
    # WRITE YOUR CODE BELOW ###################################################
    A, B = jax.jacfwd(f, (0,1))(s, u)
    ###########################################################################
    return A, B

def get_next_state(state, action):
    """ Simulate one step of nonlinear dynamics """
    h = 1 / 50

    return jnp.array([
        state[0] + h * state[3] * jnp.cos(state[2]),  # xpos 
        state[1] + h * state[3] * jnp.sin(state[2]),  # ypos
        state[2] + h * state[3] * state[4],          # theta
        state[3] + h * state[5],                     # velocity
        state[4] + h * state[6],                     # kappa
        state[5] + h * action[0],                    # accel
        state[6] + h * action[1],                   # pinch
    ])

@dataclass 
class ILQRPolicy:
    reference_state_trajectory: np.ndarray 
    reference_action_trajectory: np.ndarray
    goal_state: np.ndarray
    state_dim: int = 7
    action_dim: int = 2
    
    def __post_init__(self):
        self.Qf = 100 * np.identity(self.state_dim)
        self.Q = np.identity(self.state_dim)
        self.R = np.identity(self.action_dim)
        self.compute_policy()

    def compute_policy(self):
        L, l = tracking_lqr(
            f = jax.jit(get_next_state),
            s_goal = self.goal_state,
            s_bar = self.reference_state_trajectory,
            u_bar = self.reference_action_trajectory,
            Q = self.Q, 
            R = self.R, 
            Qf = self.Qf
        )
        self.L, self.l = L, l

    def get_action(self, state, t):
        return self.reference_action_trajectory[t] \
            + self.l[t] \
            + self.L[t] @ (state - self.reference_state_trajectory[t])

def tracking_lqr(f, s_goal, s_bar, u_bar, Q, R, Qf):
    """Compute the LQR set-point tracking solution.

    Arguments
    ---------
    f : Callable
        A function describing the discrete-time dynamics, such that
        `s[k+1] = f(s[k], u[k])`.
    s0 : numpy.ndarray
        The initial state (1-D).
    s_goal : numpy.ndarray
        The goal state (1-D).
    N : int
        The time horizon of the LQR cost function.
    Q : numpy.ndarray
        The state cost matrix (2-D).
    R : numpy.ndarray
        The control cost matrix (2-D).
    Qf : numpy.ndarray
        The terminal state cost matrix (2-D).

    Returns
    -------
    s_bar : numpy.ndarray
        A 3-D array where `s_bar[k]` is the nominal state at time step `k`,
        for `k = 0, 1, ..., N-1`
    u_bar : numpy.ndarray
        A 3-D array where `u_bar[k]` is the nominal control at time step `k`,
        for `k = 0, 1, ..., N-1`
    L : numpy.ndarray
        A 3-D array where `L[k]` is the matrix gain term of the iLQR control
        law at time step `k`, for `k = 0, 1, ..., N-1`
    l : numpy.ndarray
        A 3-D array where `l[k]` is the offset term of the iLQR control law
        at time step `k`, for `k = 0, 1, ..., N-1`
    """
    n = Q.shape[0]        # state dimension
    m = R.shape[0]        # control dimension
    eps = 0.001           # termination threshold for iLQR
    max_iters = int(1e3)  # maximum number of iLQR iterations

    # Initialize control law terms `L` and `l`
    N = u_bar.shape[0]
    L = np.zeros((N, m, n))
    l = np.zeros((N, m))

    u = np.copy(u_bar)
    s = np.copy(s_bar)


    # Linearize the dynamics at each step `k` of `(s_bar, u_bar)`
    A, B = jax.vmap(linearize, in_axes=(None, 0, 0))(f, s_bar[:-1], u_bar)
    A, B = np.array(A), np.array(B)
    # WRITE YOUR CODE BELOW ###############################################
    # Update the arrays `L`, `l`, `s`, and `u`.
    ds_bar = s_bar - s_goal[np.newaxis, :]
    V = Qf
    vbold = Qf @ ds_bar[-1]
    vplain = 0.5 * ds_bar[-1] @ Qf @ ds_bar[-1]

    for k in range(N-1, -1, -1):
        # Perform the dynamic programming update
        Btk = np.transpose(B[k])
        Atk = np.transpose(A[k])
        Mk = 0.5 * (ds_bar[k] @ Q @ ds_bar[k] + u_bar[k] @ R @ u_bar[k]) + vplain
        Muk = R @ u_bar[k] + Btk @ vbold
        Mxk = Q @ ds_bar[k] + Atk @ vbold
        Muuk = R + Btk @ V @ B[k]
        Muukinv = np.linalg.inv(Muuk)
        Muxk = Btk @ V @ A[k]
        Mxxk = Q + Atk @ V @ A[k]

        l[k] = -Muukinv @ Muk 
        L[k] = -Muukinv @ Muxk
        V = Mxxk - np.transpose(L[k]) @ Muuk @ L[k] 
        vbold = Mxk - np.transpose(L[k]) @ Muuk @ l[k]
        vplain = Mk - 0.5 * l[k] @ Muuk @ l[k] 

    return L, l



if __name__ == "__main__":
    augmentation_factor = 100

    with h5py.File('datasets/simulation_output.hdf5','r') as infile:
        sim = infile['simulation_0']['goal_0']

        initial_state = sim['state_trajectory'][0]
        goal_state = sim.attrs['goal_state']
        state_trajectory = sim['state_trajectory']
        input_trajectory = sim['input_trajectory']
        obstacle_centers = sim.attrs['obstacle_centers']
        obstacle_radii = sim.attrs['obstacle_radii']
        num_steps = sim.attrs['num_steps']

        ilqr_policy = ILQRPolicy(
            reference_state_trajectory = state_trajectory[:num_steps+1], 
            reference_action_trajectory = input_trajectory[:num_steps],
            goal_state = goal_state
        )

    with h5py.File('datasets/simulation_output_augmented.hdf5', 'w') as outfile:
        subgrp = outfile.create_group('simulation_0').create_group('goal_0')

        subgrp.attrs['goal_state'] = goal_state
        subgrp.attrs['obstacle_centers'] = obstacle_centers
        subgrp.attrs['obstacle_radii'] = obstacle_radii
        subgrp.attrs['num_steps'] = num_steps
        
        state_trajectory = subgrp.create_dataset(
            f'state_trajectory', 
            shape = (augmentation_factor * num_steps, 7), 
            dtype = 'f'
        )
        input_trajectory = subgrp.create_dataset(
            f'input_trajectory',
            shape = (augmentation_factor * num_steps, 2),
            dtype = 'f'
        )

        for t in range(num_steps):
            if t % 10 == 0: 
                print(t)
            for i in range(augmentation_factor):
                # Perturb the state by adding noise to the location and orientation
                noise = np.concatenate([
                    2 * np.random.uniform(low = -1, high = 1, size = 2), 
                    0.1 * np.random.uniform(low = -1, high =1, size = 1),
                    np.zeros(4)])
                state_perturb = state_trajectory[t] + noise 
                ilqr_action = ilqr_policy.get_action(state_perturb, t)

                state_trajectory[t * augmentation_factor + i] = state_perturb
                input_trajectory[t * augmentation_factor + i] = ilqr_action


        