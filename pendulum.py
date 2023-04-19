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

    L1 = 0.6
    L2 = 0.6

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

    M = torch.tensor([[mc+mp1+mp2, (mp1*l1+mp2*L1)*torch.cos(th1)+mp2*l2*torch.cos(th1+th2), mp2*l2*torch.cos(th1+th2)],
                      [(mp1*l1+mp2*L1)*torch.cos(th1)+mp2*l2*torch.cos(th1+th2), mp1*l1**2 + mp2*(L1**2 + 2*L1*l2*torch.cos(th2) + l2**2) + I1 + I2, mp2*l2*(l2+L1*torch.cos(th2)) + I2],
                      [mp2*l2*torch.cos(th1+th2), mp2*l2*(l2+L1*torch.cos(th2)) + I2, mp2*l2**2 + I2]]).reshape(B, 3, 3)

    C = torch.tensor([[0, -(mp1*l1+mp2*L1)*torch.sin(th1)*th1dot-mp2*l2*torch.sin(th1+th2)*th1dot, -mp2*l2*torch.sin(th1+th2)*(2*th1dot+th2dot)],
                      [0, 0, -mp2*L1*l2*torch.sin(th2)*(2*th1dot+th2dot)],
                      [0, mp2*L1*l2*torch.sin(th2)*th1dot, 0]]).reshape(B, 3, 3)

    G = torch.tensor([[0], 
                      [-(mp1*l1+mp2*L1)*g*torch.sin(th1) - mp2*l2*g*torch.sin(th1+th2)],
                      [-mp2*g*l2*torch.sin(th1+th2)]]).reshape(B, 3, 1)

    D = torch.tensor([[damp * xdot], [damp * th1dot], [damp * th2dot]]).reshape(B, 3, 1)

    U = torch.tensor([[action*500], [0], [0]], dtype=torch.float).reshape(B, 3, 1)
    
    qdot = torch.tensor([xdot, th1dot, th2dot]).reshape(B,3,1)
    
    qdotdot = torch.linalg.inv(M)@(U - C@qdot - G - D)
    
    # print("analytical M ", torch.inverse(M.reshape(3,3)))
    #print("analytical M ", M.reshape(3,3))
    #print("analytical C ", (C@qdot + G).reshape(1, 3))
    #print("analytical pos ", torch.tensor([x, th1, th2]).reshape(3,))
    #print("analytical vel ", torch.tensor([xdot, th1dot, th2dot]).reshape(3,))
    #print("analytical acc ", qdotdot.reshape(3,))
    #print("analytical D ", (-D).reshape(1, 3))

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
def rollout_dynamics(N, init_state):
    xs_nom = torch.zeros((N, 6))
    #xs_nom[0, :] = change_of_coords(init_state).reshape(6,)
    xs_nom[0, :] = init_state.reshape(6,)
    # apply a random uk to the dynamical system in open loop
    us_nom = torch.randn((N-1, 1))
    # loop through T the number of timesteps
    for t in range(N-1):
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

def linearize_dynamics(state, control):
    """
        Linearizes cartpole dynamics around linearization point (state, control). Uses autograd of analytic dynamics
    Args:
        state: torch.tensor of shape (6,) representing cartpole state
        control: torch.tensor of shape (1,) representing the force to apply

    Returns:
        A: torch.tensor of shape (6, 6) representing Jacobian df/dx for dynamics f
        B: torch.tensor of shape (6, 1) representing Jacobian df/du for dynamics f

    """
    state = torch.unsqueeze(state, 0).reshape(6,1)
    control = torch.unsqueeze(control, 0).reshape(1,1)
    
    # Physical properties

    dt = 0.05

    damp = 0.05
    g = 9.81
    mc = 10.47197551
    mp1 = 4.19873858
    mp2 = 4.19873858

    L1 = 0.6
    L2 = 0.6

    l1 = 0.3
    l2 = 0.3

    I1 = 0.15497067
    I2 = 0.15497067

    # Extract state

    x = state[0,:]
    th1 = state[1,:]
    th2 = state[2,:]
    xdot = state[3,:]
    th1dot = state[4,:]
    th2dot = state[5,:]

    # Equations of motion

    M = torch.tensor([[mc+mp1+mp2, (mp1*l1+mp2*L1)*torch.cos(th1)+mp2*l2*torch.cos(th1+th2), mp2*l2*torch.cos(th1+th2)],
                      [(mp1*l1+mp2*L1)*torch.cos(th1)+mp2*l2*torch.cos(th1+th2), mp1*l1**2 + mp2*(L1**2 + 2*L1*l2*torch.cos(th2) + l2**2) + I1 + I2, mp2*l2*(l2+L1*torch.cos(th2)) + I2],
                      [mp2*l2*torch.cos(th1+th2), mp2*l2*(l2+L1*torch.cos(th2)) + I2, mp2*l2**2 + I2]]).reshape(3, 3)

    C = torch.tensor([[0, -(mp1*l1+mp2*L1)*torch.sin(th1)*th1dot-mp2*l2*torch.sin(th1+th2)*th1dot, -mp2*l2*torch.sin(th1+th2)*(2*th1dot+th2dot)],
                      [0, 0, -mp2*L1*l2*torch.sin(th2)*(2*th1dot+th2dot)],
                      [0, mp2*L1*l2*torch.sin(th2)*th1dot, 0]]).reshape(3, 3)

    H = torch.tensor([[1*500], [0], [0]], dtype=torch.float).reshape(3, 1)

    # print(f'{torch.linalg.inv(M)=}')
    # print(f'{C=}')
    a22 = -torch.linalg.inv(M) @ C
    # print(f'{a22=}')
    A = torch.zeros((6,6))
    A[:3,3:] = torch.eye(3)
    A[3:,3:] = a22

    b2 = torch.linalg.inv(M) @ H
    # print(f'{b2=}')
    B = torch.cat((torch.zeros((3,1)), b2), dim=0)

    A = torch.eye(6) + dt * A
    B = dt * B

    return A, B


def set_state(env, pos, vel, acc):
    '''
    Set the state of the cartpole
    
    Args:    
        env: gym environment
        pos: list of length 3 representing the position of the cartpole
        vel: list of length 3 representing the velocity of the cartpole
        acc: list of length 3 representing the acceleration of the cartpole

    Returns:
        state: list of length 12 representing the state of the cartpole
    '''
    env.reset()
    env.data.qpos = pos
    env.data.qvel = vel
    env.data.qacc = acc

    return np.concatenate(
        [
            env.data.qpos[:1],  # cart x pos
            np.sin(env.data.qpos[1:]),  # link angles
            np.cos(env.data.qpos[1:]),
            np.clip(env.data.qvel, -10, 10),
            np.clip(env.data.qfrc_constraint, -10, 10),
        ]
    ).ravel()


def get_state(env):
    '''
    Get the state of the cartpole

    Args:
        env: gym environment

    Returns:
        state: list of length 12 representing the state of the cartpole
    
    '''

    return np.concatenate(
        [
            env.data.qpos[:1],  # cart x pos
            np.sin(env.data.qpos[1:]),  # link angles
            np.cos(env.data.qpos[1:]),
            np.clip(env.data.qvel, -10, 10),
            np.clip(env.data.qfrc_constraint, -10, 10),
        ]
    ).ravel()


def batched_dynamics(state, action):
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

    L1 = 0.6
    L2 = 0.6

    l1 = 0.3
    l2 = 0.3

    I1 = 0.15497067
    I2 = 0.15497067

    # Extract state

    x = state[:, 0]
    th1 = state[:, 1]
    th2 = state[:, 2]
    xdot = state[:, 3]
    th1dot = state[:, 4]
    th2dot = state[:, 5]

    # Equations of motion
    M = torch.zeros((B, 3, 3), dtype=torch.float)

    M[:, 0, 0] = mc + mp1 + mp2
    M[:, 0, 1] = (mp1*l1+mp2*L1)*torch.cos(th1)+mp2*l2*torch.cos(th1+th2)
    M[:, 0, 2] = mp2*l2*torch.cos(th1+th2)
    M[:,1, 0] = (mp1*l1+mp2*L1)*torch.cos(th1)+mp2*l2*torch.cos(th1+th2)
    M[:,1, 1] = mp1*l1**2 + mp2*(L1**2 + 2*L1*l2*torch.cos(th2) + l2**2) + I1 + I2
    M[:,1, 2] = mp2*l2*(l2+L1*torch.cos(th2)) + I2
    M[:,2, 0] = mp2*l2*torch.cos(th1+th2)
    M[:,2, 1] = mp2*l2*(l2+L1*torch.cos(th2)) + I2
    M[:,2, 2] = mp2*l2**2 + I2


    #M = torch.tensor([[mc+mp1+mp2, (mp1*l1+mp2*L1)*torch.cos(th1)+mp2*l2*torch.cos(th1+th2), mp2*l2*torch.cos(th1+th2)],
    #                  [(mp1*l1+mp2*L1)*torch.cos(th1)+mp2*l2*torch.cos(th1+th2), mp1*l1**2 + mp2*(L1 **2 + 2*L1*l2*torch.cos(th2) + l2**2) + I1 + I2, mp2*l2*(l2+L1*torch.cos(th2)) + I2],
    #                  [mp2*l2*torch.cos(th1+th2), mp2*l2*(l2+L1*torch.cos(th2)) + I2, mp2*l2**2 + I2]]).reshape(B, 3, 3)
    
    C = torch.zeros((B, 3, 3), dtype=torch.float)
    C[:, 0, 1] = -(mp1*l1+mp2*L1)*torch.sin(th1)*th1dot-mp2*l2*torch.sin(th1+th2)*th1dot
    C[:, 0, 2] = -mp2*l2*torch.sin(th1+th2)*(2*th1dot+th2dot)
    C[:, 1, 2] = -mp2*L1*l2*torch.sin(th2)*(2*th1dot+th2dot)
    C[:, 2, 1] = mp2*L1*l2*torch.sin(th2)*th1dot

    #C = torch.tensor([[0, -(mp1*l1+mp2*L1)*torch.sin(th1)*th1dot-mp2*l2*torch.sin(th1+th2)*th1dot, -mp2*l2*torch.sin(th1+th2)*(2*th1dot+th2dot)],
    #                  [0, 0, -mp2*L1*l2*torch.sin(th2)*(2*th1dot+th2dot)],
    #                  [0, mp2*L1*l2*torch.sin(th2)*th1dot, 0]]).reshape(B, 3, 3)

    G = torch.zeros((B, 3, 1), dtype=torch.float)
    G[:, 1, 0] = -(mp1*l1+mp2*L1)*g*torch.sin(th1) - mp2*l2*g*torch.sin(th1+th2)
    G[:, 2, 0] = -mp2*g*l2*torch.sin(th1+th2)

    #G = torch.tensor([[0],
    #                  [-(mp1*l1+mp2*L1)*g*torch.sin(th1) - mp2*l2*g*torch.sin(th1+th2)],
    #                  [-mp2*g*l2*torch.sin(th1+th2)]]).reshape(B, 3, 1)

    D = torch.zeros((B, 3, 1), dtype=torch.float)
    D[:, 0, 0] = damp * xdot
    D[:, 1, 0] = damp * th1dot
    D[:, 2, 0] = damp * th2dot

    #D = torch.tensor([[damp * xdot], [damp * th1dot],
    #                 [damp * th2dot]]).reshape(B, 3, 1)

    U = torch.zeros((B, 3, 1), dtype=torch.float)
    U[:, 0] = action*500

    #U = torch.tensor([[action*500], [0], [0]],
    #                 dtype=torch.float).reshape(B, 3, 1)

    qdot = torch.zeros((B, 3, 1), dtype=torch.float)
    #print(xdot.shape)
    qdot[:, 0, 0] = xdot
    qdot[:, 1, 0] = th1dot
    qdot[:, 2, 0] = th2dot


    #qdot = torch.tensor([xdot, th1dot, th2dot]).reshape(B, 3, 1)

    qdotdot = torch.linalg.inv(M)@(U - C@qdot - G - D)

    # print("analytical M ", torch.inverse(M.reshape(3,3)))
    #print("analytical M ", M.reshape(3,3))
    #print("analytical C ", (C@qdot + G).reshape(1, 3))
    #print("analytical pos ", torch.tensor([x, th1, th2]).reshape(3,))
    #print("analytical vel ", torch.tensor([xdot, th1dot, th2dot]).reshape(3,))
    #print("analytical acc ", qdotdot.reshape(3,))
    #print("analytical D ", (-D).reshape(1, 3))

    # Compute next state

    xdd = qdotdot[:, 0]
    th1dd = qdotdot[:, 1]
    th2dd = qdotdot[:, 2]

    next_xdot = xdot.reshape(B,1) + xdd*dt
    next_th1dot = th1dot.reshape(B,1) + th1dd*dt
    next_th2dot = th2dot.reshape(B, 1) + th2dd*dt


    next_x = x.reshape(B,1) + next_xdot*dt
    next_th1 = th1.reshape(B,1) + next_th1dot*dt
    next_th2 = th2.reshape(B,1) + next_th2dot*dt

    # Wrap angles

    next_th1 = torch.atan2(torch.sin(next_th1), torch.cos(next_th1))
    next_th2 = torch.atan2(torch.sin(next_th2), torch.cos(next_th2))

    next_state = torch.cat((next_x, next_th1, next_th2, next_xdot, next_th1dot, next_th2dot), 1)

    return next_state

