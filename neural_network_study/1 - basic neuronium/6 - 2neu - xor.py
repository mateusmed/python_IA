
"""
esse script tem por objetivo mostrar graficos
"""

import matplotlib.pyplot as plt
import numpy as np

# Dados de entrada (XOR)
x = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 1, 1, 0])

fig, ax = plt.subplots()

for i in range(len(x)):
    if y[i] == 0:
        ax.scatter(x[i][0], x[i][1], color='red', label='0' if i == 0 else "")
    else:
        ax.scatter(x[i][0], x[i][1], color='blue', label='1' if i == 1 else "")

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_title('Problema do XOR')
ax.legend()
ax.grid(True)


ax.plot([0.5, 0.5], [-0.5, 1.5], 'g--', label='Neuronio 1')
ax.plot([-0.5, 1.5], [0.5, 0.5], 'm--', label='Neuronio 2')

ax.legend()

plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.show()
