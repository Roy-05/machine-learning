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
   
#  

### Assignment 1: kNN Algorithm Classification

To access the program(s), clone the repository and enter the kNN directory as:

```
cd machine-learning/kNN/
```

#### Command Line Arguments:

The program will be invoked as follows:

```
knn_classify pendigits_training pendigits_test <k>
```
The arguments provide to the program the following information:  

  * The first argument, *pendigits training*, is the name of the training file with training data stored.  
  * The second argument, *pendigits test*, is the test file with the test data is stored.  
  * The third argument specifies the value of *k* for the k-nearest neighbor classifier.  
  
The training and test files will follow the same format as the text files in the [UCI datasets directory](https://archive.ics.uci.edu/ml/index.php).  

#### Implementation of kNN Algorithm  

1. Each dimension should be normalized, separately from all other dimensions. Specifically, for both training and test objects, each dimension should be transformed using
function: ```F(v) = (v−mean)/std``` , using the mean and std of the values of that dimension on
the TRAINING data. To compute the std, use the function: ```std = √(|v - mean|²/N)```  

2. Use the L2 distance (the Euclidean distance) for computing the nearest neighbors.  

#### Classification Stage  

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

### Assignment 2: Linear Regression using Gradient Descent Algorithm

To access the program(s), clone the repository and enter the Linear Regression directory as:

```
cd machine-learning/Linear\ Regression/
```  

#### Implementation of Gradient Descent Algorithm  

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
  
### Assignment 3: Perceptron Learning Algorithm  

A perceptron is able to classify linear separable dataset. To access the program(s), clone the repository and enter the Linear Regression directory as:

```
cd machine-learning/Perceptron/
```  

#### Implementation of Perceptron Learning Algorithm  

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
