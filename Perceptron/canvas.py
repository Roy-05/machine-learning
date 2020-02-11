import perceptron_learning_algorithm as pla
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from random import random

fig = plt.figure(num = "Perceptron Learning Algrotihm")

ax = plt.axes(xlim=[0, 1000], ylim=[0, 1000])

print(pla.m, pla.c)
ax.plot([0, 1000], [pla.c, pla.m*1000 + pla.c],  color = "green", lw=2)

for point in pla.dataset:
    facecolor = "k" if point[2] == 1 else "none"        #k = black
    ax.scatter(point[0], point[1], facecolor = facecolor, edgecolor='k')

plt.show()