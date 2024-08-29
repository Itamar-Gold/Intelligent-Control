import pandas as pd
import matplotlib.pyplot as plt

Freq = 50

# Read the file into a DataFrame
columns = ['Time', 'Delta_Time', 'Position', 'Reference Position', 'Input']
dfPU = pd.read_csv(f'PartiallyUnknown/continuous_process_log_{Freq}Hz_step.txt', header=None, names=columns, sep=',')
dfFU = pd.read_csv(f'fullyUnknown/continuous_process_log_{Freq}Hz_step.txt', header=None, names=columns, sep=',')
dfPID = pd.read_csv(f'continuous_process_log_{Freq}Hz_step.txt', header=None, names=columns, sep=',')


# RBAC Pi Pico Implementation - Position
plt.figure(figsize=(12, 8))
plt.plot(dfPID['Time'], dfPID['Reference Position'], label='Reference Position', linewidth=2)
plt.plot(dfPU['Time'], dfPU['Position'], label=f'Partially Unknown {Freq} hz', linewidth=2)
plt.plot(dfFU['Time'], dfFU['Position'], label=f'Fully Unknown {Freq} hz', linewidth=2)
plt.plot(dfPID['Time'], dfPID['Position'], label=f'PID {Freq} hz', linewidth=2)

plt.xlabel('t [sec]', fontsize=12)
plt.ylabel('Position', fontsize=12)
plt.legend()
plt.grid(True)
plt.title('Adaptive Control with RBAC Neural Network Vs. PID - Pi Pico Implementation')
plt.savefig(f'plots/q6/RBAC Neural Network Vs. PID - Pi Pico Implementation - Position{Freq}hz.png')
plt.close()
# plt.show()

Error_PU = abs(dfPU['Position'] - dfPU['Reference Position'])
Error_FU = abs(dfFU['Position'] - dfFU['Reference Position'])
Error_PID = abs(dfPID['Position'] - dfPID['Reference Position'])


# RBAC Pi Pico Implementation - Error
plt.figure(figsize=(12, 8))
plt.plot(dfPU['Time'], Error_PU, label=f'Partially Unknown {Freq} hz', linewidth=2)
plt.plot(dfFU['Time'], Error_FU, label=f'Fully Unknown {Freq} hz', linewidth=2)
plt.plot(dfPID['Time'], Error_PID, label=f'PID {Freq} hz', linewidth=2)
plt.xlabel('t [sec]', fontsize=12)
plt.ylabel('Error', fontsize=12)
plt.legend()
plt.grid(True)
plt.title('Adaptive Control with RBAC Neural Network Vs. PID - Pi Pico Implementation')
plt.savefig(f'plots/q6/RBAC Neural Network Vs. PID - Pi Pico Implementation - Error {Freq}hz.png')
plt.close()
# plt.show()

# RBAC Pi Pico Implementation - Input
# plt.figure(figsize=(12, 8))
# plt.plot(df1hz['Time'], df1hz['Input'], label='1 hz', linewidth=2)
# plt.plot(df5hz['Time'], df5hz['Input'], label='5 hz', linewidth=2)
# plt.plot(df10hz['Time'], df10hz['Input'], label='10 hz', linewidth=2)
# plt.plot(df20hz['Time'], df20hz['Input'], label='20 hz', linewidth=2)
# plt.plot(df50hz['Time'], df50hz['Input'], label='50 hz', linewidth=2)
# plt.xlabel('t [sec]', fontsize=12)
# plt.ylabel('Input', fontsize=12)
# plt.legend()
# plt.grid(True)
# plt.title('Adaptive Control with RBAC Neural Network - Pi Pico Implementation')
# plt.savefig(f'plots/q5/Adaptive Control with RBAC Neural Network - Pi Pico Implementation - Input.png')
# plt.close()
# plt.show()
