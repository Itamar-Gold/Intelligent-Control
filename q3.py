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
T_START, T_END, DT = 0, 50, 0.01

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
    P = -linalg.solve_continuous_lyapunov(A, np.eye(2))
    eig_P, _ = np.linalg.eig(P)
    var_list.save_lamda1(eig_P[0])
    var_list.save_lamda2(eig_P[1])

    w_dot = -GAMMA * e @ P @ B * h

    return u, w_dot, k1_hat, c1_hat, M


# 7. System Dynamics
def system_dynamics(state, t, nn):
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
    t_resh.append(t)
    u_resh.append(u.item())
    g_resh.append([K2*(x**3), C2*(x_dot*abs(x_dot)), x, x_dot])

    # Calculating The Model Dynamics
    x_ddot = (-K1 * x - K2 * x**3 - C1 * x_dot - C2 * x_dot * abs(x_dot) + u) / M

    return np.concatenate(([x_dot, x_ddot], w_dot))


# 8. Simulation Function
def run_simulation():
    nn = RadialBasisNN(2, NEURONS)
    initial_state = np.concatenate(([1, 0], nn.w))
    t = np.arange(T_START, T_END, DT)

    # Calculating The System Response
    solution = odeint(system_dynamics, initial_state, t, args=(nn,))

    return t, solution


# 9. Run Simulation and Plot Results
# Run Simulation
var_list = save_var()
t, solution = run_simulation()

# Activation Of Parameters For Plots
## Transform List To Array
g = np.array(g_resh)

## Controller constants
K_1 = 6
K_0 = round(K_1**2 / (4*M), 2)

## defining NN for 9.6-9.9
neu_num = 8 # Number of neurons chosen for the NN as a factor of number of initial parameters
c_graph = np.vstack((np.linspace(-3, 3, num=neu_num), np.linspace(-1, 1, num=neu_num)))
b_graph = np.ones(neu_num) * 0.77
w_graph = np.linspace(-2, 2, neu_num)

## Getting the values For The NN Parameters
print(f"C=", c_graph)
print(f"b=", b_graph)
print(f"W=", w_graph)

# 9.1 Plot Gaussian function
x_gas = np.arange(-10, 10, 0.1)
y_gas = []

plt.figure(figure=(15, 4))

# Graph for b
plt.subplot(1, 3, 1)

# Define a color map
colors = plt.cm.rainbow(np.linspace(0, 1, 6))

for b, color in zip([1, 2, 3, 4, 5, 6], colors):
    for i in x_gas:
        y_gas = [gaussian(np.array([i]), np.array([0]), b) for i in x_gas]
        plt.plot(x_gas, y_gas, color=color)

# Create legend entries separately
legend_elements = [plt.Line2D([0], [0], color=color, lw=2, label=f'b={b}')
                   for b, color in zip([1, 2, 3, 4, 5, 6], colors)]

plt.legend(handles=legend_elements, loc='best', bbox_to_anchor=(1, 1))
plt.title("Gaussian Function with Varying b")
plt.xlabel("x", fontsize=10)
plt.ylabel("y", fontsize=10)


#Empting y for next graph
y_gas=[]

# Graph for c
plt.subplot(1, 3, 2)

# Define a color map
colors = plt.cm.rainbow(np.linspace(0, 1, 6))

for c, color in zip([-3, -2, -1, 1, 2, 3], colors):
    for i in x_gas:
        y_gas = [gaussian(np.array([i]), c, np.array([1])) for i in x_gas]
        plt.plot(x_gas, y_gas, color=color)

# Create legend entries separately
legend_elements = [plt.Line2D([0], [0], color=color, lw=2, label=f'c={c}')
                   for c, color in zip([-3, -2, -1, 1, 2, 3], colors)]

plt.legend(handles=legend_elements, loc='best', bbox_to_anchor=(1, 1))
plt.title("Gaussian Function with Varying c")
plt.xlabel("x", fontsize=10)
plt.ylabel("y", fontsize=10)

#Empting y for next graph
y_gas=[]

# Graph for w
plt.subplot(1, 3, 3)
plt.grid()

# Define a color map
colors = plt.cm.rainbow(np.linspace(0, 1, 6))

for c, color in zip([-3, -2, -1, 1, 2, 3], colors):
    for i in x_gas:
        y_gas = [c * gaussian(np.array([i]), c, np.array([1])) for i in x_gas]
        plt.plot(x_gas, y_gas, color=color)

# Create legend entries separately
legend_elements = [plt.Line2D([0], [0], color=color, lw=2, label=f'w={c}')
                   for c, color in zip([-3, -2, -1, 1, 2, 3], colors)]

plt.legend(handles=legend_elements, loc='best', bbox_to_anchor=(1, 1))
plt.title("Gaussian Function with Varying w")
plt.xlabel("x", fontsize=10)
plt.ylabel("y", fontsize=10)

#Empting y for next graph
y_gas=[]

plt.grid()
plt.savefig(f'plots/q3/Gaussian Function.png')
plt.close()


# 9.2 Plot RBAC VS Reference
plt.figure(figsize=(12, 8))
r, _, _ = reference_signal(t)
plt.plot(t, r, label='Reference', linewidth=2)
plt.plot(t, solution[:, 0], label='System Output', linewidth=2)
plt.xlabel('t [Sec]', fontsize=12)
plt.ylabel('Position', fontsize=12)
plt.legend()
plt.grid(True)
plt.title('Adaptive Control with RBAC Neural Network')
plt.savefig(f'plots/q3/RBAC VS Reference.png')
plt.close()

# 9.3 Plot Tracking Error
plt.figure(figsize=(12, 4))
plt.plot(t, abs(r - solution[:, 0]), label='Tracking Error')
plt.xlabel('t [Sec]', fontsize=12)
plt.ylabel('Error', fontsize=12)
plt.legend()
plt.grid(True)
plt.title('Tracking Error')
plt.savefig(f'plots/q3/Tracking Error.png')
plt.close()

# 9.4 Error Performance Check
Err = np.abs(r - solution[:, 0]).mean()
print(f"Mean Error For RBAC With Unknown Nonlinear Functions: {Err}")

# 9.4 Plot Weights Performance
plt.figure(1, figsize=(16,6))
row , col = int((solution.shape[1] - 2) / 2) + 1, 2
for i in range(solution.shape[1] -2):
    plt.subplot(row, col, 1+i)
    plt.plot(t, solution[:, 2+i], label=f"$w_{i}$, x(t), $k_0={K_0}, k_1={K_1}$")
    plt.legend(prop={'size':12}, framealpha=1)
    plt.xlabel("t [Sec]", fontsize=12)
    plt.ylabel("w", fontsize=12)
    plt.grid()
plt.savefig(f'plots/q3/Weights Performance.png')
plt.close()

# 9.5 Plot Approximated g(t) VS g(t)
plt.figure(1, figsize=(16,6))
plt.xlabel("t [Sec]", fontsize=12)
plt.ylabel("$g(t)$", fontsize=12)
#g = np.array(g_resh)
plt.plot(g[:, 0] + g[:, 1], label=f"g(t), $k_0={K_0}, k_1={K_1}$")
plt.plot(g_hat_resh, label="$\hat{g}(t)$"+ f", $k_0={K_0}, k_1={K_1}$")
plt.legend(prop={'size':17}, framealpha=1, loc='upper right')
plt.grid()
plt.savefig(f'plots/q3/Approximated g(t) VS g(t).png')
plt.close()

# 9.6 Gaussian Function VS Position
x = np.arange(-5, 5,0.01)
y = []

# Plot
plt.figure(figsize=(16,7))
plt.xlabel("x [m]", fontsize=12)
plt.ylabel("$g(t)$", fontsize=12)
for s, c_s in enumerate(c_graph[0, :]):
    for j in x:
        y.append(w_graph[s]*gaussian(j, c_s, b_graph[0]))
    w_new = w_graph[s]
    plt.plot(x, y, label=f"$c_{s}={round(c_s, 2)}, w_{s}={round(w_new, 2)}$")
    y=[]
plt.legend()
plt.grid()
plt.savefig(f'plots/q3/Gaussian Function VS Position.png')
plt.close()

# 9.7 Plot Value Range Of The Position & The Nonlinear Function VS x(t)
x = np.linspace(-1,1,100)
x_dot = np.linspace(-0.6, 0.6, 100)
y = []
x_97 = 0
plt.xlabel('x [m]', fontsize=12)
plt.ylabel('$g(t)$', fontsize=12)
for s in x:
    for i, c_i in enumerate(c_graph[0, :]):
        x_97 += w_graph[i] * gaussian(s, c_i, b_graph[0])
    y.append(x_97)
    x_97 = 0
plt.plot(x, y, label='$\hat{g}(0)$')
plt.scatter(g[:, 2], g[:, 0] + g[:, 1], s=2, c="r", label='$g(t)$')
plt.grid()
plt.legend()
plt.savefig(f'plots/q3/Value Range Of The Position & The Nonlinear Function VS x(t).png')
plt.close()

# 9.8 Gaussian Function VS Speed
x = np.arange(-5, 5, 0.01)
y = []

# Plot
plt.figure(figsize=(16,7))
plt.xlabel("$\dot{x}$ [m/2]", fontsize=12)
plt.ylabel("$g(t)$", fontsize=12)
for s, c_s in enumerate(c_graph[1, :]):
    for j in x:
        y.append(w_graph[s]*gaussian(j, c_s, b_graph[0]))
    w_new = w_graph[s]
    plt.plot(x, y, label=f"$c_{s}={round(c_s, 2)}, w_{s}={round(w_new, 2)}$")
    y=[]
plt.legend()
plt.grid()
plt.savefig(f'plots/q3/Gaussian Function VS Speed.png')
plt.close()

# 9.9 Plot Value Range Of The Position & The Nonlinear Function VS Speed
x = np.linspace(-1.7,1.7,100)
x_dot = np.linspace(-0.6, 0.6, 100)
y = []
x_99 = 0
g = np.array(g_resh)
plt.xlabel('$\dot{x}$ [m/2]', fontsize=12)
plt.ylabel('$g(t)$', fontsize=12)
for s in x:
    for i, c_i in enumerate(c_graph[1, :]):
        x_99 += w_graph[i] * gaussian(s, c_i, b_graph[0])
    y.append(x_99)
    x_99 = 0
plt.plot(x, y, label='$\hat{g}(0)$')
plt.scatter(g[:, 3], g[:, 0] + g[:, 1], s=2, c="r", label='$g(t)$')
plt.grid()
plt.legend()
plt.savefig(f'plots/q3/Value Range Of The Position & The Nonlinear Function VS Speed.png')
plt.close()

# 9.10 g(t) & Path Of The State Vector

# Calculation of the nonlinear function g(t)
def calc_nonlinear(x_calc, x_dot_calc):
    f = K2*x_calc**3 + C2*x_dot_calc*abs(x_dot_calc)
    return f

# Activation
x_910= np.linspace(-1, 1, 30)
x_dot_910 = np.linspace(-1.5, 1.5, 30)
x_mesh, x_dot_mesh = np.meshgrid(x_910, x_dot_910)

G = calc_nonlinear(x_mesh,x_dot_mesh)

# Plot
plt.figure(figsize=(10,7))
ax = plt.axes(projection='3d')
ax.scatter(g[:, 2], g[:, 3], g[:, 0] + g[:, 1], c="r")
ax.contour(x_mesh, x_dot_mesh, G, 50, cmap='binary')
ax.set_zlabel('$g(x, \dot{x})$', fontsize=12)
ax.set_xlabel('$x$', fontsize=12)
ax.set_ylabel('$\dot{x}$', fontsize=12)
plt.title('Nonlinear Function Visualization')
plt.savefig(f'plots/q3/Nonlinear Function Visualization.png')
plt.close()

hashh = np.linspace(0, len(var_list.m), len(var_list.m))
actual_k1 = np.ones(len(var_list.m)) * M
m_max_list = np.ones(len(var_list.m)) * 0.9
# RBAC variables prediction plot
plt.figure(figsize=(12, 8))
plt.plot(hashh, var_list.k1, label='$k_{1}$', linewidth=2)
plt.plot(hashh, var_list.c1, label='$c_{1}$', linewidth=2)
plt.plot(hashh, var_list.m, label='m', linewidth=2)
plt.plot(hashh, m_max_list, '--k', label='$m_{max}$', linewidth=2)
plt.xlabel('t [sec]', fontsize=12)
plt.ylabel('variable value', fontsize=12)
plt.legend()
plt.grid(True)
plt.title('Adaptive Control with RBAC Neural Network - Variables prediction')
plt.savefig(f'plots/q3/Adaptive Control with RBAC Neural Network - Variables prediction.png')
plt.close()

# RBAC variables prediction plot
plt.figure(figsize=(12, 8))
plt.plot(hashh, var_list.lamda1, label='$\lambda_{1}$', linewidth=2)
plt.plot(hashh, var_list.lamda2, label='$\lambda_{2}$', linewidth=2)
plt.xlabel('t [sec]', fontsize=12)
plt.ylabel('variable value', fontsize=12)
plt.legend()
plt.grid(True)
plt.title('Adaptive Control with RBAC Neural Network - P Eigenvalues')
plt.savefig(f'plots/q3/Adaptive Control with RBAC Neural Network - P Eigenvalues.png')
plt.close()

print(f'Initial guess of the NN is: -- m = {var_list.m[0]} -- k1 = {var_list.k1[0]} -- c1 = {var_list.c1[0]}')