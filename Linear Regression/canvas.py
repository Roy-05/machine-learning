import regressor as rg
import matplotlib.pyplot as plt
from random import random


start_x = 0
start_y = rg.b

end_x = max(rg.x_points)
end_y = end_x * rg.w + rg.b

#plt.plot([start_x,end_x], [start_y, end_y])
plt.plot(rg.x_points, rg.y_points, 'ro')

w1 = random()
b1 = random()

for i in range(rg.epochs):
    dW, dB = rg.get_accumulated_errors(w1, b1)
    w1 += (2/rg.datapoints) * dW * rg.learning_rate
    b1 += (2/rg.datapoints) * dB * rg.learning_rate


    start_x = 0
    start_y = b1

    end_x = max(rg.x_points)
    end_y = end_x * w1 + b1

    plt.plot([start_x,end_x], [start_y, end_y])

plt.show()
