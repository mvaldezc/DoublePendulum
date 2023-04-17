import numpy as np
import torch

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

    dt = 0.05

    damp = 0.05
    g = 9.81
    mc = 10.47197551
    mp1 = 4.19873858
    mp2 = 4.19873858

    L1 = 0.92 # got by solving from M matrix given by mujoco (makes no sense lol but makes M matrix the same thing as mujoco)
    L2 = 0.92

    l1 = 0.3
    l2 = 0.3

    I1 = 0.15497067
    I2 = 0.15497067

    # Extract state

    x = state[:,0]
    th1 = state[:,1]
    th2 = state[:,2]
    xdot = state[:,3]
    th1dot = state[:,4]
    th2dot = state[:,5]

    # Equations of motion
    
    M = torch.tensor([[mc+mp1+mp2, (mp1*l1+mp2*L1)*torch.cos(th1), mp2*l2*torch.cos(th2)],
                  [(mp1*l1+mp2*L1)*torch.cos(th1), mp1*l1**2 + mp2*L1**2 + I1, mp2*L1*l2*torch.cos(th1-th2)],
                  [mp2*l2*torch.cos(th2), mp2*L1*l2*torch.cos(th1-th2), mp2*l2**2 + I2]], dtype=torch.float64).reshape(B, 3, 3)

    C = torch.tensor([[0, -(mp1*l1+mp2*L1)*torch.sin(th1)*th1dot, -mp2*l2*torch.sin(th2)*th2dot],
                  [0, 0, mp2*L1*l2*torch.sin(th1-th2)*th2dot],
                  [0, -mp2*L1*l2*torch.sin(th1-th2)*th1dot, 0]], dtype=torch.float64).reshape(B, 3, 3)
    
    G = torch.tensor([[0], [-(mp1*l1+mp2*L1)*g*torch.sin(th1)], [-mp2*g*l2*torch.sin(th2)]], dtype=torch.float64).reshape(B, 3, 1)
    
    D = torch.tensor([[damp * xdot], [damp * th1dot], [damp * th2dot]], dtype=torch.float64).reshape(B, 3, 1)

    U = torch.tensor([[action], [0], [0]], dtype=torch.float64).reshape(B, 3, 1)
    
    qdot = torch.tensor([xdot, th1dot, th2dot], dtype=torch.float64).reshape(3,1)

    qdotdot = torch.linalg.inv(M)@(U - C@qdot - G - D)

    #print("analytical", M.reshape(3,3))
    #print("analytical", (C@qdot + G + D).reshape(1,3))
    
    # Compute next state

    xdd = qdotdot[:,0]
    th1dd = qdotdot[:,1]
    th2dd = qdotdot[:,2]

    next_xdot = xdot + xdd*dt
    next_th1dot = th1dot + th1dd*dt
    next_th2dot = th2dot + th2dd*dt

    next_x = x + next_xdot*dt
    next_th1 = th1 + next_th1dot*dt
    next_th2 = th2 + next_th2dot*dt

    # Wrap angles
    next_th1 = torch.atan2(torch.sin(next_th1), torch.cos(next_th1))
    next_th2 = torch.atan2(torch.sin(next_th2), torch.cos(next_th2))

    next_state = torch.cat((next_x, next_th1, next_th2, next_xdot, next_th1dot, next_th2dot), 1)

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

    th1 = np.arctan2(sin_th1_mujoco, cos_th1_mujoco)
    th2 = np.arctan2(sin_th2_mujoco, cos_th2_mujoco)

    return torch.tensor([x, th1, th2, xdot, th1dot, th2dot]) 

def T_change_of_coords(state): 
    x = state[:,0].reshape(-1,1)
    sin_th1_mujoco = state[:,1].reshape(-1,1)
    sin_th2_mujoco = state[:,2].reshape(-1,1)
    cos_th1_mujoco = state[:,3].reshape(-1,1)
    cos_th2_mujoco = state[:,4].reshape(-1,1)
    xdot = state[:,5].reshape(-1,1)
    th1dot = state[:,6].reshape(-1,1)
    th2dot = state[:,7].reshape(-1,1)

    th1 = np.arctan2(sin_th1_mujoco, cos_th1_mujoco).reshape(-1,1)
    th2 = np.arctan2(sin_th2_mujoco, cos_th2_mujoco).reshape(-1,1)
    
    return np.concatenate((x, th1, th2, xdot, th1dot, th2dot), axis=1)

## rollout dynamics to obtain an N-length nominal trajectories of states and control inputs
    # T is the trajectory length in timsteps
def rollout_dynamics(T, init_state):
    xs_nom = torch.zeros((T+1, 6))
    xs_nom[0, :] = change_of_coords(init_state).reshape(6,)
    # apply a random uk to the dynamical system in open loop
    us_nom = torch.zeros((T, 1))
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
    state = torch.unsqueeze(state, 0)
    control = torch.unsqueeze(control, 0)
    J = torch.autograd.functional.jacobian(dynamics_analytic, (state, control))
    A = J[0].reshape((6, 6))
    B = J[1].reshape((6, 1))

    return A, B
