# Machine Learning - CSC 736

This repository stores my projects and work for CSC 736: Machine Learning, at Missouri State University

## Contents:  
 1. [Assignment 1: kNN Algorithm Classification](#assignment-1-knn-algorithm-classification)  
    * [Command Line Arguments](#command-line-arguments)  
    * [Implementation of kNN Algoritm](#implementation-of-knn-algorithm)  
    * [Classification Stage](#classification-stage)  
   
 2. [Assignment 2: Linear Regression using Gradient Descent Algorithm](#assignment-2-linear-regression-using-gradient-descent-algorithm) 
    * [Implementation of Gradient Descent Algoritm](#implementation-of-gradient-descent-algorithm)  
   
 3. [Assignment 3: Perceptron Learning Algorithm](#assignment-3-perceptron-learning-algorithm)  
    * [Implementation of Perceptron Learning Algorithm](#implementation-of-perceptron-learning-algorithm)  
    
 4. [Assignment 4: Artificial Neural Network](#assignment-4-artificial-neural-network)
    * [The Data](#the-data)
    * [Training](#training)
    * [Testing](#testing)
   

## Assignment 1: kNN Algorithm Classification

To access the program, clone the repository and enter the kNN directory as:

```
cd machine-learning/kNN/
```

### Command Line Arguments:

The program will be invoked as follows:

```
knn_classify pendigits_training pendigits_test <k>
```
The arguments provide to the program the following information:  

  * The first argument, *pendigits training*, is the name of the training file with training data stored.  
  * The second argument, *pendigits test*, is the test file with the test data is stored.  
  * The third argument specifies the value of *k* for the k-nearest neighbor classifier.  
  
The training and test files will follow the same format as the text files in the [UCI datasets directory](https://archive.ics.uci.edu/ml/index.php).  

### Implementation of kNN Algorithm  

1. Each dimension should be normalized, separately from all other dimensions. Specifically, for both training and test objects, each dimension should be transformed using
function: ```F(v) = (v−mean)/std``` , using the mean and std of the values of that dimension on
the TRAINING data. To compute the std, use the function: ```std = √(|v - mean|²/N)```  

2. Use the L2 distance (the Euclidean distance) for computing the nearest neighbors.  

### Classification Stage  

For each test object you should print a line containing the following info:
* **object ID** - This is the line number where that object occurs in the test file. Start with
0 in numbering the objects, not with 1.
* **predicted class** (the result of the classification) - If your classification result is a tie
among two or more classes, choose one of them randomly.
* **true class** (from the last column of the test file)
* **accuracy** - This is defined as follows:  

  * If there were no ties in your classification result, and the predicted class is correct,
the accuracy is 1.  

  * If there were no ties in your classification result, and the predicted class is incorrect, the accuracy is 0.  
  
  * If there were ties in your classification result, and the correct class was one of the
classes that tied for best, the accuracy is 1 divided by the number of classes that
tied for best.  

  * If there were ties in your classification result, and the correct class was NOT one
of the classes that tied for best, the accuracy is 0. 

## Assignment 2: Linear Regression using Gradient Descent Algorithm

To access the program, clone the repository and enter the Linear Regression directory as:

```
cd machine-learning/Linear\ Regression/
```  

### Implementation of Gradient Descent Algorithm  

A linear regressor is able to predict values with given inputs based on provided training dataset. In this assignment, develop a program that is able to:  
  1. Generate points in the training set.
        * Arbitrarily define a line `y = wx + b (eg. y = 2x+3)` as your ground truth line.  
        * Generate 20 random data points (randomly select 20 x and calculate 20 y accordingly) from the line defined above. The y value on each point needs to randomly add or minus a noise with the range of `10% ∗ y`. For example, assuming your line is `y = 2x + 3`. Your first point is `x = 10, y = 23 + random(23 ∗ 0.1) or y = 23 − random(23 ∗ 0.1)`.  
        * Visualize the line in green and the 20 points (filled circles) on a graphic user interface.  
  2. Implement a linear regression with the gradient descent learning algorithm.
        * Randomly initialize the weight and bias to a double within (0, 1).  
        * Set your learning rate *η = 0.00001*
        * Train your linear regressor by the gradient descent learning algorithm with the provided training data generated from the previous step.  
  3. Visualize the line represented by the current weights at the end of each epoch on
GUI. 

  4. Train your linear regression model for at least 50 epochs. 
  
## Assignment 3: Perceptron Learning Algorithm  

A perceptron is able to classify linear separable dataset. To access the program, clone the repository and enter the Linear Regression directory as:

```
cd machine-learning/Perceptron/
```  

### Implementation of Perceptron Learning Algorithm  

1. Generate points in the training set:    

   * Arbitrarily define a line (eg. y = ax+b or ax+by+c=0)  
   * Generate 20 random data points on a 1000 by 1000 size canvas. Based on the line in the previous step, assign the class (1 or -1) to each points.
   * Visualize the line in green and the points (circles filled or unfilled) on a graphic user interface.
  
2. Implement a perceptron and perceptron learning algorithm:

   * Randomly initialize the weights to a double within (0, 1).  
   * Set your learning rate to 0.0001  
   * Train your perceptron by the perceptron learning algorithm with the provided training data generated from the previous step.  
   * Define “epoch” as one iteration of training all the training data one time.  
   * Visualize the line represented by the current weights at the end of each epoch on GUI.    
   * Output the number of misclassification on the training data at the end of each epoch.  
   * Terminate the training process if all the training data are correctly classified by the perceptron.  
   
## Assignment 4: Artificial Neural Network  

The purpose of this assignment is to implement a simple three layer neural network (input layer, hidden layer, and output layer) with the Backpropagation learning algorithm on a handwritten digits recognition problem. To access the program, clone the repository and enter the kNN directory as:

```
cd machine-learning/ANN/
```

### The Data  

* The data file, called optdigits-3.tra, is an ASCII file containing 1535 examples, one per line. Each example is a comma-separated (without any white space) list of 65 integer values, the first 64 specifying the input and the last value specifying the digit which is the desired output. 

* The input values are integers in the range [0..16]. 

* You should first normalize the input values by converting them to reals in the range [0..1] by dividing every value by 16.0. This is useful because the derivative of the sigmoid function is often very close to 0, which can cause the network to converge very slowly to a good set of weights.  

* You’ll have to convert the desired output digit to a target output vector for the four output units (0, 1, 2, 3). For example, if the digit is a “3” then create the target vector [0.1 0.1 0.1 0.9]. Using this set of teacher output values is preferred because the sigmoid function cannot produce the exact output values of 0 and 1 using finite weights, and so the weight values may get very, very large causing overflow problems.  

### Training

* You are to implement a validation set to evaluate the performance of your network during the training process. To do this, you should divide the input file into two parts, each containing 80% and 20% data samples. For the training process, you will use the 80% of the training data as the training set and the remaining 20% will be the validation set. 

* After every 10 epochs compute the mean squared error (MSE) on the training set and validation set.  

* You should compute this MSE value of your training examples after every 10 epochs by stopping backpropogation and running through all of the training examples (not validation examples) to compute this error value. You should not compute this “on the fly”
after each training example is used to update the weights because then the error for each example would be based on a different network (i.e., set of weights).  

* You should not use your validation examples to tune your weights in the network.  

### Testing 

* Test your network using the examples in the test set. Report the percentage correct classification on the test set.  

* Define the output digit computed by the network as the corresponding output unit with maximum activation (i.e., output) value.  
