from random import randint, random
import tkinter

x_points = []
y_points = []
w = randint(1, 5)
b = randint(1, 10)
datapoints = 20
epochs = 2000
learning_rate = 0.000001

for i in range(datapoints):

    x_points.append(randint(1,50))


for x in x_points:

    # y = w * x + b
    # Yn = y +/- randint(0.1 * y)
    y = w * x + b
    y_points.append(y + (-1)**randint(0,1) * randint(0,round(0.1 * y)))

w1 = random()
b1 = random()

def update_weight_and_bias(w, b):
    w_derive = 0
    b_derive = 0
    for i in range(datapoints):
        w_derive += x_points[i] * (y_points[i] - (w * x_points[i] + b))
        b_derive += y_points[i] - (w * x_points[i] + b)
    
    w += (w_derive / float(datapoints)) * learning_rate
    b += (b_derive / float(datapoints)) * learning_rate

    return w,b

top = tkinter.Tk()
canvas_width = 500
canvas_height = 500

canvas = tkinter.Canvas(top, height=canvas_width, width=canvas_height)

start_x = min(x_points)
start_y = start_x * w + b

end_x = 100
end_y = 100 * w + b

canvas.create_line(start_x, 
                canvas_height - start_y, 
                end_x, 
                canvas_height - end_y, fill = "#ff5500")

def create_circle(x, y, r, canvasName): 
    x0 = x - r
    y0 = y - r
    x1 = x + r
    y1 = y + r
    return canvasName.create_oval(x0, y0, x1, y1)

for i in range(datapoints):
    create_circle(x_points[i], canvas_height - y_points[i], 3, canvas)

# for i in range(epochs):
#     w1, b1 = update_weight_and_bias(w1, b1)

#     start_y = start_x * w1 + b1
#     end_y = 100 * w1 + b1

#     canvas.create_line(start_x, 
#                     canvas_height - start_y, 
#                     end_x, 
#                     canvas_height -end_y)


canvas.pack()
top.mainloop()