import numpy as np
import matplotlib.pyplot as plt

upper_bound = 0.9
lower_bound = 0
x = np.linspace(-10, 10, 100)
logistic = ((upper_bound - lower_bound) / (1 + np.exp(-x))) + lower_bound


# Sigmoid plot with upper and lower bound
plt.figure(figsize=(12, 8))
plt.plot(x, np.ones(len(x)) * upper_bound, label='$U_{b}$', linewidth=3)
plt.plot(x, np.ones(len(x)) * lower_bound, label='$L_{b}$', linewidth=3)
plt.plot(x, logistic, label=r'$\frac{U_{b}-L_{b}}{1+e^{-x}}+L_{b}$', linewidth=3)
plt.xlabel('x', fontsize=14)
plt.ylabel('y(x)', fontsize=14)
plt.legend(fontsize=18)
plt.grid(True)
plt.title('Sigmoid activation function modified with limits')
plt.savefig(f'plots/q3/Sigmoid activation function modified with limits.png')
plt.close()
