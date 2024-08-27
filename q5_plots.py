import pandas as pd
import matplotlib.pyplot as plt

# Read the file into a DataFrame
columns = ['Time', 'Delta_Time', 'Position', 'Reference Position', 'Input']
df1hz = pd.read_csv('continuous_process_log_1Hz_step.txt', header=None, names=columns, sep=',')
df5hz = pd.read_csv('continuous_process_log_10Hz_step.txt', header=None, names=columns, sep=',')
df10hz = pd.read_csv('continuous_process_log_10Hz_step.txt', header=None, names=columns, sep=',')
df20hz = pd.read_csv('continuous_process_log_10Hz_step.txt', header=None, names=columns, sep=',')
df50hz = pd.read_csv('continuous_process_log_50Hz_step.txt', header=None, names=columns, sep=',')


# RBAC Pi Pico Implementation - Position
plt.figure(figsize=(12, 8))
plt.plot(df1hz['Time'], df1hz['Reference Position'], label='Reference Position', linewidth=2)
plt.plot(df1hz['Time'], df1hz['Position'], label='1 hz', linewidth=2)
plt.plot(df5hz['Time'], df5hz['Position'], label='5 hz', linewidth=2)
plt.plot(df10hz['Time'], df10hz['Position'], label='10 hz', linewidth=2)
plt.plot(df20hz['Time'], df20hz['Position'], label='20 hz', linewidth=2)
plt.plot(df50hz['Time'], df50hz['Position'], label='50 hz', linewidth=2)
plt.xlabel('t [sec]', fontsize=12)
plt.ylabel('Position', fontsize=12)
plt.legend()
plt.grid(True)
plt.title('Adaptive Control with RBAC Neural Network - Pi Pico Implementation')
plt.savefig(f'plots/q5/Adaptive Control with RBAC Neural Network - Pi Pico Implementation - Position.png')
plt.close()
# plt.show()

Error_1hz = abs(df1hz['Position'] - df1hz['Reference Position'])
Error_5hz = abs(df5hz['Position'] - df5hz['Reference Position'])
Error_10hz = abs(df10hz['Position'] - df10hz['Reference Position'])
Error_20hz = abs(df20hz['Position'] - df20hz['Reference Position'])
Error_50hz = abs(df50hz['Position'] - df50hz['Reference Position'])

# RBAC Pi Pico Implementation - Error
plt.figure(figsize=(12, 8))
plt.plot(df1hz['Time'], Error_1hz, label='1 hz', linewidth=2)
plt.plot(df5hz['Time'], Error_5hz, label='5 hz', linewidth=2)
plt.plot(df10hz['Time'], Error_10hz, label='10 hz', linewidth=2)
plt.plot(df20hz['Time'], Error_20hz, label='20 hz', linewidth=2)
plt.plot(df50hz['Time'], Error_50hz, label='50 hz', linewidth=2)
plt.xlabel('t [sec]', fontsize=12)
plt.ylabel('Error', fontsize=12)
plt.legend()
plt.grid(True)
plt.title('Adaptive Control with RBAC Neural Network - Pi Pico Implementation')
plt.savefig(f'plots/q5/Adaptive Control with RBAC Neural Network - Pi Pico Implementation - Error.png')
plt.close()
# plt.show()

# RBAC Pi Pico Implementation - Input
plt.figure(figsize=(12, 8))
plt.plot(df1hz['Time'], df1hz['Input'], label='1 hz', linewidth=2)
plt.plot(df5hz['Time'], df5hz['Input'], label='5 hz', linewidth=2)
plt.plot(df10hz['Time'], df10hz['Input'], label='10 hz', linewidth=2)
plt.plot(df20hz['Time'], df20hz['Input'], label='20 hz', linewidth=2)
plt.plot(df50hz['Time'], df50hz['Input'], label='50 hz', linewidth=2)
plt.xlabel('t [sec]', fontsize=12)
plt.ylabel('Input', fontsize=12)
plt.legend()
plt.grid(True)
plt.title('Adaptive Control with RBAC Neural Network - Pi Pico Implementation')
plt.savefig(f'plots/q5/Adaptive Control with RBAC Neural Network - Pi Pico Implementation - Input.png')
plt.close()
# plt.show()
