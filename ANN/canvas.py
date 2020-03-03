import ANN as ann
import matplotlib.pyplot as plt
import numpy as np



fig = plt.figure(num = "MSE for Backpropagation Algorithm")


y = ann.training_mse

ax = plt.axes(xlim=[0,len(y)], ylim=[min(y)-0.01, max(y)+0.01])

i = 0
for point in ann.training_mse:     
    ax.scatter(i, point, facecolor='red')
    i += 1

j = 0
for point in ann.validation_mse:
    ax.scatter(j, point, facecolor='blue')
    j += 1

plt.show()