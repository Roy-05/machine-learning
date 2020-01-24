# Machine Learning - CSC 736

This repository stores my projects and work for CSC 736: Machine Learning, at Missouri State University

## Assignment 1: kNN Algorithm Implementation

To run the program, clone the repository and enter the kNN directory as:

```
cd machine-learning/kNN
```

### Command Line Arguments:

The program will be invoked as follows:

```
knn_classify pendigits_training pendigits_test <k>
```
The arguments provide to the program the following information: The first argument, pendigits training, is the name of the training file with training data stored. The second argument, pendigits test, is the test file with the test data is stored. The third argument
specifies the value of k for the k-nearest neighbor classifier. The training and test files will follow the same format as the text files in the UCI datasets directory. A description of the datasets and the file format can be found in the folder. For each dataset, a training file and a test file are provided. The name of each file indicates what dataset the file belongs to, and whether the file contains training or test data.  

### Implementation Guidelines  

1. Each dimension should be normalized, separately from all other dimensions. Specifically, for both training and test objects, each dimension should be transformed using
function ```F(v) = (v−mean)/std``` , using the mean and std of the values of that dimension on
the TRAINING data. To compute the std, using function ```std = √(|v - mean|²/N)```  

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
