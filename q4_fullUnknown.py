import numpy as np
from scipy.integrate import odeint
from scipy import linalg
import matplotlib.pyplot as plt
import torch


# Table Of Contents
## lines 11-24: Activation of the system
## lines 24-113: Main Function
## lines 116-137: Activation of parameters for graphs
## lines 139-369: Graphs

# Final Vir = 0.5, 15, 1.5, 6, 0.75, 6, 0.05
# Final NN = c=(-3,3),(-1,1), b=0.77, w=(-2,2)


# 1. Constants and System Parameters
NEURONS = 8
M, K1, K2, C1, C2 = 0.5, 15, 1.5, 6, 0.75
C_sys, GAMMA = 6, 0.05
T_START, T_END, DT = 0, 10, 0.01

# 2. Lists of data saving for graphs
t_resh, u_resh, g_resh, g_hat_resh= [], [], [], []
w_resh = []


# 3. Activation Function
def gaussian(x, c, b):
    return np.exp(-np.linalg.norm(x - c)**2 / (2*b**2))


# 4. RadialBasisNN Class
class RadialBasisNN:
    def __init__(self, num_inputs, num_neurons):
        self.c = np.vstack((np.linspace(-3, 3, num=num_neurons), np.linspace(-1, 1, num=num_neurons)))
        self.b = np.ones(num_neurons) * 0.77
        self.w = np.linspace(-2, 2, num_neurons)

    def predict(self, x):
        h = np.array([gaussian(x, self.c[:, i], self.b[i]) for i in range(len(self.b))])
        return np.dot(self.w, h), h


class VarRadialBasisNN:
    def __init__(self, num_inputs, num_neurons, width):
        self.c = np.vstack((np.linspace(-30, 30, num=num_neurons), np.linspace(-15, 15, num=num_neurons)))
        self.b = np.ones(num_neurons) * width
        self.w = np.linspace(-10, 10, num=num_neurons)

    def predict(self, x):
        h = np.array([gaussian(x, self.c[:, i], self.b[i]) for i in range(len(self.b))])
        return abs(np.dot(self.w, h))


class MRadialBasisNN:
    def __init__(self, num_inputs, num_neurons, width):
        self.c = np.vstack((np.linspace(0, 1, num=32), np.linspace(0.2, 0.8, num=32)))
        self.b = np.ones(num_neurons) * width
        self.w = np.linspace(0, 0.2, num=32)

    def log_predict(self, x, m_max):
        h = np.array([gaussian(x, self.c[:, i], self.b[i]) for i in range(len(self.b))])
        logistic = m_max / (1 + np.exp(-np.dot(self.w, h)))
        return logistic


class save_var:
    def __init__(self):
        self.k1 = []
        self.c1 = []
        self.m = []
        self.lamda1 = []
        self.lamda2 = []
        self.x_fullyUnknown = []
        self.x_semiUnknown = []
        self.x_ref = []

    def save_k1(self, x):
        self.k1.append(x)

    def save_c1(self, x):
        self.c1.append(x)

    def save_m(self, x):
        self.m.append(x)

    def save_lamda1(self, x):
        self.lamda1.append(x)

    def save_lamda2(self, x):
        self.lamda2.append(x)

    def save_x_fullyUnknown(self, x):
        self.x_fullyUnknown.append(x)

    def save_x_semiUnknown(self, x):
        self.x_semiUnknown.append(x)

    def save_x_ref(self, x):
        self.x_ref.append(x)


# 5. Reference Signal
def reference_signal(t, step=False):
    if step:
        t = np.array(t)
        r = np.ones(t.shape)
        r_dot = np.zeros(t.shape)
        r_ddot = np.zeros(t.shape)
    t = torch.tensor(t, dtype=torch.float32, requires_grad=True)
    r = torch.sin(0.5 * t).requires_grad_(True)
    r_dot = torch.autograd.grad(r, t, torch.ones_like(r), create_graph=True)[0]
    r_ddot = torch.autograd.grad(r_dot, t, torch.ones_like(r_dot), create_graph=True)[0]
    return r.detach().numpy(), r_dot.detach().numpy(), r_ddot.detach().numpy()


# 6. Controller
def controller(x, x_dot, r, r_dot, r_ddot, nn, k1_nn, c1_nn, m_nn):
    # Predict variables
    k1_hat = k1_nn.predict(np.array([x, x_dot]))
    c1_hat = c1_nn.predict(np.array([x, x_dot]))
    M = m_nn.log_predict(np.array([x, x_dot]), 0.9)

    # save variables
    var_list.save_k1(k1_hat)
    var_list.save_c1(c1_hat)
    var_list.save_m(M)

    print(f'-- k1 = {k1_hat}  -- c1 = {c1_hat} -- m = {M}')

    # Calculating the input
    q = C_sys ** 2 / (4 * M)
    K = np.array([q, C_sys])
    e = np.array([r - x, r_dot - x_dot])

    g_hat, h = nn.predict(np.array([x, x_dot]))

    # Saving Data For Graph
    g_hat_resh.append([g_hat.item()])

    u = (k1_hat * x + g_hat + c1_hat * x_dot + np.dot(K, e) + M * r_ddot)

    # Calculating W dot using Lyapunov
    B = np.array([[0, 1/M]]).T
    K_0 = K[0]
    K_1 = K[1]
    A = np.array([[0, 1], [-K_0/M, -K_1/M]])
    P = -linalg.solve_discrete_lyapunov(A, np.eye(2))
    eig_P, _ = np.linalg.eig(P)
    var_list.save_lamda1(eig_P[0])
    var_list.save_lamda2(eig_P[1])

    w_dot = -GAMMA * e @ P @ B * h

    return u, w_dot, k1_hat, c1_hat, M


# 7. System Dynamics
def system_dynamics(state, t, nn, dt):
    x, x_dot = state[:2]
    w = state[2:]

    r, r_dot, r_ddot = reference_signal(t)
    k1_nn = VarRadialBasisNN(2, 64, 5)
    c1_nn = VarRadialBasisNN(2, 64, 4)
    m_nn = MRadialBasisNN(2, 32, 1)

    c1_nn.c = np.vstack((np.linspace(-10, 10, num=64), np.linspace(-5, 5, num=64)))
    c1_nn.w = np.linspace(-5, 5, num=64)

    nn.w = w  # Update NN weights
    u, w_dot, K1, C1, M = controller(x, x_dot, r, r_dot, r_ddot, nn, k1_nn, c1_nn, m_nn)

    # Saving Data For Graphs
    g_resh.append([K2*(x**3), C2*(x_dot*abs(x_dot)), x, x_dot])

    # Calculating The Model Dynamics
    x_ddot = (-K1 * x - K2 * x**3 - C1 * x_dot - C2 * x_dot * abs(x_dot) + u) / M
    nn.w = w + w_dot * dt
    x_dot = x_dot + x_ddot * dt

    return np.concatenate(([x_dot, x_ddot], w_dot))


# 8. Simulation Function
def run_simulation(frequency):
    nn = RadialBasisNN(2, NEURONS)
    initial_state = np.concatenate(([1, 0], nn.w))
    t = np.arange(T_START, T_END, DT)
    discrete_time = 1 / frequency
    continuous_time = DT
    num_steps = int(discrete_time / continuous_time)
    solution = []
    discrete_list = []
    u = []

    current_state = initial_state
    for i, time in enumerate(t):
        if i % num_steps == 0:
            discrete_list.append(time)
            solution.append(current_state[:2])

            # Call controller and update u_resh here
            r, r_dot, r_ddot = reference_signal(time)
            k1_nn = VarRadialBasisNN(2, 64, 5)
            c1_nn = VarRadialBasisNN(2, 64, 4)
            m_nn = MRadialBasisNN(2, 32, 1)
            u, _, _, _, _ = controller(current_state[0], current_state[1], r, r_dot, r_ddot, nn, k1_nn, c1_nn, m_nn)

            # Integrate over small time step
            sol = odeint(system_dynamics, current_state, [time, time + DT], args=(nn, DT))
            current_state = sol[-1]  # Update current state
        u_resh.append(u)

    return t, np.array(discrete_list), np.array(solution)


# 9. Run Simulation and Plot Results
# plt.figure(figsize=(12, 8))
# i = 0
# for frequencie in [10, 20, 50, 80]:
#     var_list = save_var()
#     t, discrete_list, solution = run_simulation(frequencie)
#     r, _, _ = reference_signal(t)
#     if i == 0:
#         plt.plot(t, r, label='Reference', linewidth=2)
#     i += 1
#     plt.plot(discrete_list, solution[:, 0], label=f'{frequencie} hz', linewidth=2)
# plt.xlabel('t [Sec]', fontsize=12)
# plt.ylabel('Position', fontsize=12)
# plt.legend()
# plt.grid(True)
# plt.title('Adaptive Control with RBAC - Performance')
# plt.savefig(f'plots/q4/Performance_fullUnknown.png')
# plt.close()
#
#
# plt.figure(figsize=(12, 8))
# for frequencie in [10, 20, 50, 80]:
#     var_list = save_var()
#     t, discrete_list, solution = run_simulation(frequencie)
#     r, _, _ = reference_signal(discrete_list)
#     plt.plot(discrete_list, abs(r - solution[:, 0]), label=f'{frequencie} hz', linewidth=2)
# plt.xlabel('t [Sec]', fontsize=12)
# plt.ylabel('Error', fontsize=12)
# plt.legend()
# plt.grid(True)
# plt.title('Discrete Adaptive Control with RBAC - Error')
# plt.savefig(f'plots/q4/Error_fullUnknown.png')
# plt.close()

# u_dis = []
# t_dis = []
# for index, value in enumerate(t_resh):
#     for discrete_time in discrete_list:
#         if value == discrete_time:
#             t_dis.append(value)
#             u_dis.append(u_resh[index])

# plt.figure(figsize=(12, 8))
# for frequencie in [10, 20, 50, 80]:
#     var_list = save_var()
#     t, discrete_list, solution = run_simulation(frequencie)
#     plt.plot(t, u_resh, label=f'{frequencie} hz', linewidth=2)
#     t = []
#     u_resh = []
# plt.xlabel('t [Sec]', fontsize=12)
# plt.ylabel('Input', fontsize=12)
# plt.legend()
# plt.grid(True)
# plt.title('Discrete Adaptive Control with RBAC - Input')
# plt.savefig(f'plots/q4/Input_fullUnknown.png')
# plt.close()
