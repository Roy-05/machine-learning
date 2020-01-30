
let x_points = [],
    y_points = [],
    w = randInt(1, 5),
    b = randInt(1, 10),
    datapoints = 20,
    epochs = 50,
    learning_rate = 0.0001;

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

canvas.width = 600;
canvas.height = 600;      


function generateDataPoints(){
    for(let i=0; i<datapoints; i++){
        x_points.push(randInt(1,50));
        
        y = w * x_points[i] + b;
        y_points.push(y + (-1)**randInt(0,1) * randInt(0,Math.round(0.1 * y)));
    }
}

generateDataPoints()


let start_x = Math.min(...x_points),
    start_y = start_x * w + b,
    end_x = Math.max(...x_points),
    end_y = end_x * w + b,
    w1 = Math.random(),
    b1 = Math.random(),
    points = [];

drawLine(ctx, start_x, end_x, start_y, end_y, "green");

for(let i=0; i<epochs; i++){

    let dW = get_accumulated_errors(w1, b1)[0],
        dB = get_accumulated_errors(w1, b1)[1]

    w1 += (2/datapoints) * dW * learning_rate
    b1 += (2/datapoints) * dB * learning_rate

    end_x = Math.max(...x_points)
    end_y = end_x * w1 + b1

    points.push({
        "x0": start_x,
        "x1": end_x,
        "y0": start_y,
        "y1": end_y
    });

}

points.forEach(point => {
    // drawLine(ctx, point["x0"],point["x1"], point["y0"], point["y1"], "red")
});


function drawLine(ctx, x0, x1, y0, y1, color) {
    ctx.lineWidth = 2
    ctx.beginPath();       
    ctx.moveTo(x0, canvas.height - y0);    
    ctx.lineTo(x1, canvas.height - y1);  
    ctx.strokeStyle = color
    ctx.stroke();      
}

function get_accumulated_errors(w, b) {

    let dW = 0,
        dB = 0;

    for(let i=0; i<datapoints; i++){

        dW += x_points[i] * (y_points[i] - (w * x_points[i] + b));
        dB += y_points[i] - (w * x_points[i] + b);
    }

    return [dW, dB]

}

function randInt(min, max) {
    min = Math.ceil(min);
    max = Math.floor(max);
    return Math.floor(Math.random() * (max - min + 1)) + min; 
    }