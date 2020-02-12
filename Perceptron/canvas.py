import perceptron_learning_algorithm as pla
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from random import random

fig = plt.figure(num = "Perceptron Learning Algrotihm")

ax = plt.axes(xlim=[0, 1000], ylim=[0, 1000])

ax.plot([0, 1000], [pla.c, pla.m*1000 + pla.c],  color = "green", lw=3, zorder = 1)

for point in pla.dataset:
    #k = black
    facecolor = "k" if point[3] == 1 else "none"        
    ax.scatter(point[1], point[2], facecolor = facecolor, edgecolor='k', zorder = 2)


[line] = ax.plot([],[],lw=3, color = "red", zorder=1)

def animate(i):
    if(i>=len(pla.points)):
        plt.pause(1)
        plt.close('all')
    else:
        line.set_data(pla.points[i][0], pla.points[i][1])

anim = FuncAnimation(
        fig, 
        animate, 
        frames = pla.epochs+1, 
        interval = 250)

plt.show()