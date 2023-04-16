import numpy as np

def dynamics_analytic(state, action):
    """
        Computes x_t+1 = f(x_t, u_t) using analytic model of dynamics in Pytorch
        Should support batching
    Args:
        state: torch.tensor of shape (B, 4) representing the cartpole state
        control: torch.tensor of shape (B, 1) representing the force to apply

    Returns:
        next_state: torch.tensor of shape (B, 4) representing the next cartpole state

    """
    next_state = None
    dt = 0.05
    g = 9.81
    mc = 1
    mp1 = 0.1
    mp2 = 0.1
    l1 = 0.5
    l2 = 0.5

    # --- Your code here
    # Transform mujoco state into global reference frame (generalized coordinates)
    x = state[:, 0]
    sin_th1_mujoco = state[:, 1]
    sin_th2_mujoco = state[:, 2]
    cos_th1_mujoco = state[:, 3]
    cos_th2_mujoco = state[:, 4]
    xdot = state[:, 5]
    th1dot = state[:, 6]
    th2dot = state[:, 7]

    th1 = np.arctan2(state[:, 3]-state[:, 1])
    th2 = np.arctan2(state[:, 4]-state[:, 2])

    # Equations of motion

    M = np.array([[mc+mp1+mp2, l1*(mp1*mp2)*np.cos(th1), mp2*l2*np.cos(th2)],
                    [l1*(mp1*mp2)*np.cos(th1), l1**2*(mp1+mp2), l1*l2*mp2*np.cos(th1-th2)],
                    [mp2*l2*np.cos(th2), l1*l2*mp2*np.cos(th1-th2), l2**2*mp2]])
    
    C = np.array([[l1*(mp1*mp2)*np.sin(th1)*th1dot**2 + l2*mp2*np.sin(th2)*th2dot**2],
                  [-l1*l2*mp2*np.sin(th1-th2)*th2dot**2 + g*l1*(mp1+mp2)*np.sin(th1),],
                    [l1*l2*mp2*np.sin(th1-th2)*th1dot**2 + g*l2*mp2*np.sin(th2)]])

    U = np.array([[action],[0],[0]])

    F = C + U

    statedot = np.linalg.inv(M)@F
    
    # Compute next state

    xdd = statedot[0]
    th1dd = statedot[1]
    th2dd = statedot[2]

    next_xdot = xdot + xdd*dt
    next_th1dot = th1dot + th1dd*dt
    next_th2dot = th2dot + th2dd*dt

    next_x = x + next_xdot*dt
    next_th1 = th1 + next_th1dot*dt
    next_th2 = th2 + next_th2dot*dt

    next_state = np.concatenate((next_x, next_th1, next_th2, next_xdot, next_th1dot, next_th2dot), axis=1)

    # ---

    return next_state


def change_of_coords(state):
    x = state[:, 0]
    sin_th1_mujoco = state[:, 1]
    sin_th2_mujoco = state[:, 2]
    cos_th1_mujoco = state[:, 3]
    cos_th2_mujoco = state[:, 4]
    xdot = state[:, 5]
    th1dot = state[:, 6]
    th2dot = state[:, 7]

    th1 = np.arctan2(state[:, 3]-state[:, 1])
    th2 = np.arctan2(state[:, 4]-state[:, 2])

    return np.concatenate((x, th1, th2, xdot, th1dot, th2dot), axis=1)