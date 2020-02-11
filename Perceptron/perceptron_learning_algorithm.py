import random
import math

#Slope ranges from  PI/12 - 5PI/12
m = random.uniform(math.pi/12, 5*math.pi/12)
c = random.randint(-50,50)

dataset = []

for i in range(20):
    x = random.randint(50,950)
    y = random.randint(50,950)
    pt_class = 1 if y - m*x - c >= 0 else 0
    dataset.append([x, y, pt_class])
    


