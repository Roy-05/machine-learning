import regressor as rg
import matplotlib.pyplot as plt
from random import random


start_x = 0
start_y = rg.b

end_x = max(rg.x_points)
end_y = end_x * rg.w + rg.b

plt.plot([start_x,end_x], [start_y, end_y])
plt.plot(rg.x_points, rg.y_points, 'ro')
plt.show()

#     rg.w1, rg.b1 = rg.update_weight_and_bias(rg.w1, rg.b1)
