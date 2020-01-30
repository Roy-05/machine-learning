import regressor as rg
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from random import random


start_x = min(rg.x_points) 
start_y = start_x * rg.w + rg.b

end_x = max(rg.x_points)
end_y = end_x * rg.w + rg.b

w1 = random()
b1 = random()
points = []

fig = plt.figure()
ax = plt.axes(
    xlim = (min(rg.x_points) - 5, max(rg.x_points) + 5), 
    ylim = (min(min(rg.y_points),end_x*w1+b1) -5 , max(rg.y_points) + 5))

ax.plot([start_x,end_x], [start_y, end_y], color = "green", lw = 2)
ax.plot(rg.x_points, rg.y_points, 'ro')

[line] = ax.plot([],[],lw=3)

for i in range(rg.epochs):

    dW, dB = rg.get_accumulated_errors(w1, b1)
    w1 += (2/rg.datapoints) * dW * rg.learning_rate
    b1 += (2/rg.datapoints) * dB * rg.learning_rate

    points.append([[start_x,end_x], [start_y, end_y]])
    end_y = end_x * w1 + b1


def update(i):
    if(i>=len(points)):
        plt.pause(1)
        plt.close('all')
    
    line.set_data(points[i][0], points[i][1])

anim = FuncAnimation(fig, update, interval = 20, frames=200)

plt.show()
