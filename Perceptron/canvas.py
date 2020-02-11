import perceptron_learning_algorithm as pla
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


fig = plt.figure(num = "Perceptron Learning Algrotihm")

ax = plt.axes(xlim=[0, 1000], ylim=[0, 1000])

print(pla.m, pla.c)
ax.plot([0, 1000], [pla.c, pla.m*1000 + pla.c], color = "green", lw=2)

plt.show()