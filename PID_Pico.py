import machine
from machine import Timer
from ulab import numpy as np
import time
import sys


def get_datetime():
    datetime = time.gmtime()
    return f"{datetime[3]}-{datetime[4]}-{datetime[5]}__{datetime[0]}-{datetime[1]}-{datetime[2]}"


def reference_signal(t):
    r = np.sin(0.5 * t)
    r_dot = 0.5 * np.cos(0.5 * t)
    r_ddot = -0.25 * np.sin(0.5 * t)
    return r, r_dot, r_ddot


def continuous_process(timer):
    global test_log
    global log_row
    global last_enter_time_cp
    global x, u
    global x_dot

    # Get current time and dt
    t = (time.ticks_ms() - t_ref) / 1000
    dt = t - last_enter_time_cp

    # Model dynamics
    x_ddot = -(k1 / m) * x - (k2 / m) * (x ** 3) - (c1 / m) * x_dot - (c2 / m) * (x_dot * abs(x_dot)) + u / m

    # Integration
    x_dot = x_dot + x_ddot * dt
    x = x + x_dot * dt

    (r, r_dot, r_ddot) = reference_signal(t)

    # Write performance to the continuous log
    continuous_process_log.write(f"{t}, {dt}, {x}, {r}, {u}\n")

    last_enter_time_cp = t


def controller(timer):
    global last_enter_time_c, u, k_p, e, k_d, e_dot, k_i, e_integral

    # Calculate actual dt
    t = (time.ticks_ms() - t_ref) / 1000
    dt = t - last_enter_time_c

    # Get current reference signal
    (r, r_dot, r_ddot) = reference_signal(t)

    # Calculate error
    e = r - x
    e_dot = r_dot - x_dot
    e_dot = e_dot + e * dt
    e_integral = e + e * dt
    print(f"e: {e}, e_dot: {e_dot}, e_integral: {e_integral}")

    # PID control
    u = k_p * e + k_d * e_dot + k_i * e_integral

    # Write current time and dt to the controller log
    controller_log.write(f"{t}, {dt}\n")
    print(f"x: {x}, r: {r}")
    last_enter_time_c = t


def stop_simulation(t):
    continuous_timer.deinit()
    controller_timer.deinit()

    print("Write logs to file...")
    continuous_process_log.close()
    controller_log.close()

    print("Simulation finished")

# Main code

# Model constants
global u, k_p, e, k_d, e_dot, k_i, e_integral
k1, k2, c1, c2, m = 15, 1.5, 6, 0.75, 0.5
k_p = 35
k_d = 10
k_i = 2
e = 0
e_dot = 0
e_integral = 0

# Set controller frequency
controller_rate = input("Enter controller rate: ")
controller_rate = int(controller_rate)

# Initial Conditions - System model
x, x_dot, u = 0, 0, 0
t_ref = time.ticks_ms()
last_enter_time_c = 0
last_enter_time_cp = 0

# Set log files

# Controller log to monitor time and dt
controller_log_filename = f"controller_log_{controller_rate}Hz_step.txt"
controller_log = open(controller_log_filename, "w")

# Real time performance monitor
continuous_process_log_filename = f"continuous_process_log_{controller_rate}Hz_step.txt"
continuous_process_log = open(continuous_process_log_filename, "w")

# Set timer for the simulation
stopwatch = Timer()
stopwatch.init(mode=Timer.ONE_SHOT, period=1000 * 20, callback=stop_simulation)

# Start simulation
print("Simulation is running...")
continuous_timer = Timer()
controller_timer = Timer()
controller_timer.init(period=int(1000.0 / controller_rate), mode=Timer.PERIODIC, callback=controller)
continuous_timer.init(period=10, mode=Timer.PERIODIC, callback=continuous_process)
