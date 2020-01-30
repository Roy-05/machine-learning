import regressor as rg
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from random import random
from time import sleep


start_x = min(rg.x_points) - 10 
start_y = start_x * rg.w + rg.b

end_x = max(rg.x_points)
end_y = end_x * rg.w + rg.b



w1 = random()
b1 = random()
points = []

for i in range(rg.epochs):
    # plt.clf()
    dW, dB = rg.get_accumulated_errors(w1, b1)
    w1 += (2/rg.datapoints) * dW * rg.learning_rate
    b1 += (2/rg.datapoints) * dB * rg.learning_rate

    start_x = min(rg.x_points)-10
    start_y = start_x * rg.w + rg.b

    end_x = max(rg.x_points)
    end_y = end_x * w1 + b1

    points.append([[start_x,end_x], [start_y, end_y]])
    plt.plot([start_x,end_x], [start_y, end_y])
    
plt.show()
