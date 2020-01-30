import regressor as rg
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from random import random


start_x = min(rg.x_points) 
start_y = start_x * rg.w + rg.b

end_x = max(rg.x_points)
end_y = end_x * rg.w + rg.b

fig = plt.figure()
ax = plt.axes(
    xlim = (min(rg.x_points) - 10, max(rg.x_points) + 10), 
    ylim = (min(rg.y_points) - 10, max(rg.y_points) + 10))

ax.plot([start_x,end_x], [start_y, end_y], color = "green", lw = 2)
ax.plot(rg.x_points, rg.y_points, 'ro')

[line] = ax.plot([],[],lw=3)



w1 = random()
b1 = random()
points = []

for i in range(rg.epochs):
    dW, dB = rg.get_accumulated_errors(w1, b1)
    w1 += (2/rg.datapoints) * dW * rg.learning_rate
    b1 += (2/rg.datapoints) * dB * rg.learning_rate

    end_y = end_x * w1 + b1

    points.append([[start_x,end_x], [start_y, end_y]])

def update(i):
    line.set_data(points[i][0], points[i][1])

anim = FuncAnimation(fig, update)

plt.show()
