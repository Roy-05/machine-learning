import ANN as ann
import matplotlib.pyplot as plt
import numpy as np



fig = plt.figure(num = "MSE for Backpropagation Algorithm")

x = np.linspace(0,10,10)
y = ann.mse_points

ax = plt.axes(xlim=[0,len(y)], ylim=[min(y)-0.01, max(y)+0.01])

i = 0
for point in ann.mse_points:     
    ax.scatter(i, point, facecolor='red')
    i +=1

z = np.polyfit(x,y,3)
f = np.poly1d(z)
x_new = np.linspace(x[0], x[-1], 50)
y_new = f(x_new)

plt.plot(x_new,y_new)

# plt.plot(x, mse_points)
plt.show()