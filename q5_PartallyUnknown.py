import machine
from machine import Timer
from ulab import numpy as np
import time
import sys

# Define utilities function 
def np_vstack(top_array, buttom_array):
    return np.array([top_array, buttom_array])

def gaussian(x, c, b):
    return np.exp(-np.linalg.norm(x - c) ** 2 / (2 * b ** 2))

def get_datetime():
    datetime = time.gmtime()
    return f"{datetime[3]}-{datetime[4]}-{datetime[5]}__{datetime[0]}-{datetime[1]}-{datetime[2]}"

class RadialBasisNN:
    def __init__(self, num_inputs, num_neurons):
        self.c = np_vstack(np.linspace(-3, 3, num=num_neurons), np.linspace(-1, 1, num=num_neurons))
        self.b = np.ones(num_neurons) * 0.77
        self.w = np.linspace(-2, 2, num_neurons)

    def predict(self, x):
        h = np.array([gaussian(x, self.c[:, i], self.b[i]) for i in range(len(self.b))])
        return np.dot(self.w, h), h


def reference_signal(t):
    r = np.sin(0.5*t)
    r_dot = 0.5*np.cos(0.5*t)
    r_ddot = -0.25*np.sin(0.5*t)
    return r, r_dot, r_ddot

def controller_RBAC(x, x_dot, r, r_dot, r_ddot, k1, c1, m, radial_basis_nn):
    E = np.array([[r - x, r_dot - x_dot]]).T
    system = np.array([x, x_dot])
    (g_hat, h) = radial_basis_nn.predict(system)
    
    q = p**2 / (4*m) * 0.99
    K = np.array([[q, p]]).T
    B = np.array([[0, 1/m]]).T
    A = np.array([[0, 1], [-K[0, 0]/m, -K[1, 0]/m]])
    
    u = k1 * x + g_hat + c1 * x_dot + np.dot(K.T, E) + m * r_ddot
    w_dot = - gamma * np.dot(np.dot(E.T, P), B) * h
    
    return u[0, 0], w_dot


def continuous_process(timer):
    global test_log, log_row, last_enter_time_cp, x, x_dot, dt
    
    t = (time.ticks_ms() - t_ref) / 1000
    dt = t - last_enter_time_cp
    
    # Model dynamics
    x_ddot = -(k1 / m) * x - (k2 / m) * (x ** 3) - (c1 / m) * x_dot - (c2 / m) * (x_dot * abs(x_dot)) + u / m
    
    # Integration
    x_dot = x_dot + x_ddot * dt
    x = x + x_dot * dt 
    
    (r, r_dot, r_ddot) = reference_signal(t)     
    continuous_process_log.write(f"{t}, {dt}, {x}, {r}, {u}\n")
    
    last_enter_time_cp = t

def controller(timer):
    global last_enter_time_c
    global u
    t = (time.ticks_ms() - t_ref) / 1000
    dt = t - last_enter_time_c
    
    (r, r_dot, r_ddot) = reference_signal(t)     
    (u, w_dot) = controller_RBAC(x, x_dot, r, r_dot, r_ddot, k1, c1, m, radial_basis_nn)
    radial_basis_nn.w = radial_basis_nn.w + w_dot.reshape((8,)) * dt
    
    
    controller_log.write(f"{t}, {dt}\n")
    print(f"x: {x}, r: {r}")
    last_enter_time_c = t
    
    
# Model constants
k1, k2, c1, c2, m = 15, 1.5, 6, 0.75, 0.5

# Adaptation Law Parameters
gamma = 0.1
p = 6
q = p**2 / (4*m) * 0.99
K = np.array([[q, p]]).T
B = np.array([[0, 1/m]]).T
A = np.array([[0, 1], [-K[0, 0]/m, -K[1, 0]/m]])
P = np.array([[-1.708, 0.014],
               [0.014 , -0.043 ]])

# Initial NN parameters
neurons_number = 8
# Sin parameters

global k1_nn, c1_nn, m_nn
radial_basis_nn = RadialBasisNN(2, neurons_number)

controller_rate = input("Enter controller rate: ")
controller_rate = int(controller_rate)

# Initial Conditions - System model
x, x_dot, u = 0, 0, 0
t_ref = time.ticks_ms()
last_enter_time_c = 0
last_enter_time_cp = 0

# Init log files
controller_log_filename  = f"controller_log_{controller_rate}Hz_step.txt"
controller_log = open(controller_log_filename, "w")

continuous_process_log_filename  = f"continuous_process_log_{controller_rate}Hz_step.txt"
continuous_process_log = open(continuous_process_log_filename, "w")


def stop_simulation(t):
    continuous_timer.deinit()
    controller_timer.deinit()
    
    print("Write logs to file...")    
    continuous_process_log.close()
    controller_log.close()
    
    print("Simulation finished")
    
stopwatch = Timer()
stopwatch.init(mode=Timer.ONE_SHOT, period=1000*20, callback=stop_simulation)


print("Simulation is running...")
continuous_timer = Timer()
controller_timer = Timer()
controller_timer.init(period=int(1000.0 / controller_rate), mode=Timer.PERIODIC, callback=controller)
continuous_timer.init(period=10, mode=Timer.PERIODIC, callback=continuous_process)
