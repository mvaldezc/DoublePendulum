import numpy as np
import torch
import autograd.numpy as np
from autograd import grad, jacobian

def dynamics_analytic(state, action):
    """
        Computes x_t+1 = f(x_t, u_t) using analytic model of dynamics in Pytorch
        Should support batching
    Args:
        state: torch.tensor of shape (B, 6) representing the double pendulum state
        control: torch.tensor of shape (B, 1) representing the force to apply

    Returns:
        next_state: torch.tensor of shape (B, 6) representing the next double pendulum state

    """
    B = state.shape[0]

    # Physical properties
    dt = 0.01
    #g = 9.81
    #mc = 1
    #mp1 = 0.1
    #mp2 = 0.1
    #l1 = 0.5
    #l2 = 0.5
    
    damp = 0.05
    g = -9.81
    mc = 10.47197551
    mp1 = 4.19873858
    mp2 = 4.19873858
    l1 = 0.3
    l2 = 0.3

    # Extract state

    x = state[:,0]
    th1 = state[:,1]
    th2 = state[:,2]
    xdot = state[:,3]
    th1dot = state[:,4]
    th2dot = state[:,5]

    # Equations of motion

    M = np.array([[mc+mp1+mp2, l1*(mp1*mp2)*np.cos(th1), mp2*l2*np.cos(th2)],
                    [l1*(mp1*mp2)*np.cos(th1), l1**2*(mp1+mp2), l1*l2*mp2*np.cos(th1-th2)],
                    [mp2*l2*np.cos(th2), l1*l2*mp2*np.cos(th1-th2), l2**2*mp2]], dtype=np.float64).reshape(B,3,3)
    
    C = np.array([[l1*(mp1*mp2)*np.sin(th1)*th1dot**2 + l2*mp2*np.sin(th2)*th2dot**2],
                  [-l1*l2*mp2*np.sin(th1-th2)*th2dot**2    + g*l1*(mp1+mp2)*np.sin(th1) ],
                  [l1*l2*mp2*np.sin(th1-th2)*th1dot**2 + g*l2*mp2*np.sin(th2)]], dtype=np.float64).reshape(B, 3, 1)
    
    #G = np.array([[0], [g*l1*(mp1+mp2)*np.sin(th1)], [g*l2*mp2*np.sin(th2)]], dtype=np.float64).reshape(B,3,1)
    
    D = np.array([[damp * xdot], [damp * th1dot], [damp * th2dot]], dtype=np.float64).reshape(B, 3, 1)

    U = np.array([[action], [0], [0]], dtype=np.float64)

    #F = C + G - D + U
    F = C - D + U

    #statedot = np.linalg.inv(M.reshape(3,3))@F.reshape(3,1)
    #statedot = statedot.reshape(B,3,1)
    statedot = np.linalg.inv(M)@F
    #statedot = torch.bmm(torch.inverse(torch.from_numpy(M)), torch.from_numpy(F)).numpy()
    
    # Compute next state

    xdd = statedot[:,0]
    th1dd = statedot[:,1]
    th2dd = statedot[:,2]

    next_xdot = xdot + xdd*dt
    next_th1dot = th1dot + th1dd*dt
    next_th2dot = th2dot + th2dd*dt

    next_x = x + next_xdot*dt
    next_th1 = th1 + next_th1dot*dt
    next_th2 = th2 + next_th2dot*dt

    next_state = np.concatenate((next_x, next_th1, next_th2, next_xdot, next_th1dot, next_th2dot), axis=1)

    return next_state


def change_of_coords(state): 
    x = state[0]
    sin_th1_mujoco = state[1]
    sin_th2_mujoco = state[2]
    cos_th1_mujoco = state[3]
    cos_th2_mujoco = state[4]
    xdot = state[5]
    th1dot = state[6]
    th2dot = state[7]

    #th1 = np.arctan2(cos_th1_mujoco, sin_th1_mujoco)
    #th2m = np.arctan2(sin_th2_mujoco, cos_th2_mujoco)
    th1 = np.arctan2(sin_th1_mujoco, cos_th1_mujoco)
    th2m = np.arctan2(sin_th2_mujoco, cos_th2_mujoco)
    #th2 = th2m + th1
    return np.array([x, th1, th2m, xdot, th1dot, th2dot]) 

def T_change_of_coords(state): 
    x = state[:,0].reshape(-1,1)
    sin_th1_mujoco = state[:,1].reshape(-1,1)
    sin_th2_mujoco = state[:,2].reshape(-1,1)
    cos_th1_mujoco = state[:,3].reshape(-1,1)
    cos_th2_mujoco = state[:,4].reshape(-1,1)
    xdot = state[:,5].reshape(-1,1)
    th1dot = state[:,6].reshape(-1,1)
    th2dot = state[:,7].reshape(-1,1)

    #th1 = np.arctan2(state[:,3],state[:,1]).reshape(-1,1)
    #th2 = np.arctan2(state[:,2],state[:,4]).reshape(-1,1)

    #th1 = np.arctan2(cos_th1_mujoco, sin_th1_mujoco).reshape(-1,1)
    th1 = np.arctan2(sin_th1_mujoco, cos_th1_mujoco).reshape(-1,1)
    #th2m = np.arctan2(sin_th2_mujoco, cos_th2_mujoco).reshape(-1,1)
    th2m = np.arctan2(sin_th2_mujoco, cos_th2_mujoco).reshape(-1,1)
    th2 = th2m + th1
    
    return np.concatenate((x, th1, th2m, xdot, th1dot, th2dot), axis=1)
    #return np.concatenate((sin_th1_mujoco, sin_th2_mujoco, th2, cos_th1_mujoco, cos_th2_mujoco, th2dot), axis=1)

## rollout dynamics to obtain an N-length nominal trajectories of states and control inputs
    # T is the trajectory length in timsteps
def rollout_dynamics(T, init_state):
    xs_nom = np.zeros((T+1, 6))
    xs_nom[0, :] = change_of_coords(init_state).reshape(6,)
    # apply a random uk to the dynamical system in open loop
    us_nom = np.zeros((T, 1))
    # loop through T the number of timesteps
    for t in range(T):
        curr_state = xs_nom[t, :].reshape(1,6)
        curr_action = us_nom[t, :].reshape(1,1)
        xs_nom[t+1, :] = dynamics_analytic(curr_state, curr_action)

    return xs_nom, us_nom

def linearize_pytorch(state, control):
    """
        Linearizes cartpole dynamics around linearization point (state, control). Uses autograd of analytic dynamics
    Args:
        state: torch.tensor of shape (6,) representing cartpole state
        control: torch.tensor of shape (1,) representing the force to apply

    Returns:
        A: torch.tensor of shape (6, 6) representing Jacobian df/dx for dynamics f
        B: torch.tensor of shape (6, 1) representing Jacobian df/du for dynamics f

    """
    # state = torch.from_numpy(state)
    # control = torch.from_numpy(control)
    state = state.reshape(1,6)
    control = control.reshape(1,1)
    # state = torch.unsqueeze(state, 0)
    # control = torch.unsqueeze(control, 0)
    # J = torch.autograd.functional.jacobian(dynamics_analytic, (state, control))
    # J = torch.autograd.functional.jacobian(dynamics_analytic, (state.detach().cpu().numpy(), control.detach().cpu().numpy()))
    J = jacobian(dynamics_analytic, (state, control))
    # J = jacobian(dynamics_analytic)
    print(J)
    A = J[0].reshape((6, 6))
    B = J[1].reshape((6, 1))


    return A, B
