import perceptron_learning_algorithm as pla
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from random import random

fig = plt.figure(num = "Perceptron Learning Algrotihm")

ax = plt.axes(xlim=[0, 1000], ylim=[0, 1000])

print(pla.m, pla.c)
ax.plot([0, 1000], [pla.c, pla.m*1000 + pla.c],  color = "green", lw=2)

for point in pla.dataset:
    facecolor = "k" if point[3] == 1 else "none"        #k = black
    ax.scatter(point[1], point[2], facecolor = facecolor, edgecolor='k')

 
for i in range(5):
    row = pla.dataset[0]
    x1 = row[1]
    x2 = row[2]
    y2 = pla.weights[2]*row[2] + pla.weights[0]
    y1 = pla.weights[1]*row[1] + pla.weights[0]

    m = (y2-y1)/(x2-x1)

    y0 = y1 - m * x1
    x0 = 0
    xn = 900
    yn = m * (900 - x1)  + y1

    ax.plot([x0,xn], [y0, yn])
    for row in pla.dataset:
        e = pla.feedForward(row)
        pla.train(row)
    
        print(f"Expect: {row[-1]:4} Actual: {e:4}\t{row[-1]==e}")
    print("\n\n")


plt.show()