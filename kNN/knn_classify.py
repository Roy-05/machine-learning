from sys import argv
from math import sqrt
from random import choice

def getMeanAndStd(dataset):
    """
    Accepts a 2-d array representing the dataset.
    Returns the mean and std for each column in the dataset.
    """
    meanAndStd = []
    for i in range(len(dataset[0])-1):
        column = [row[i] for row in dataset]
        mean = sum(column)/len(column)
        sigma = 0
        for datapoint in column:
            sigma += abs((datapoint - mean))**2
        
        std = sqrt(sigma/len(column))
        meanAndStd.append({"mean": mean, "std": std})

    return meanAndStd

def normalizeData(meanAndStd, dataset):
    """
    Normalizes each datapoint using the function
            f(v) = (v - mean)/std
    """

    for i in range(len(dataset)):
        for j in range(len(dataset[i])-1):
            mean = meanAndStd[j]["mean"]
            std = meanAndStd[j]["std"]
            dataset[i][j] = (dataset[i][j] - mean)/std

def euclidianDistance(row1, row2):
    """
    Returns the Euclidian Distance between two vectors
    """

    dist = 0.0
    for i in range(len(row1)-1):
        dist += (row1[i] - row2[i])**2
    
    return sqrt(dist)

def getNeighbors(training_data, test_row, k):
    """
    Returns k nearest neighbors to a vector on
    the basis of the shortest Euclidian Distance
    """

    distances = list()
    for training_row in training_data:
        dist = euclidianDistance(training_row, test_row)
        distances.append([training_row, dist])
    
    #Sort on the basis of dist
    distances.sort(key=lambda row:row[1])

    neighbors = list()

    for i in range(int(k)):
        neighbors.append(distances[i][0])

    return neighbors

def predictClass(training_data, test_row, k):
    """
    Returns the predicted class on the basis
    of the classes of the k nearest neighbors
    """

    neighbors = getNeighbors(training_data, test_row, k)
    output_vals = [row[-1] for row in neighbors]
    
    counts = dict()

    for i in output_vals:
        counts[i] = counts.get(i, 0) + 1

    v = [value for value in counts.values()]

    #Pick a class on random if ties occur
    prediction = choice([key for key in counts if counts[key] == max(v)])

    return prediction

def getCounts(training_data, test_row, k):
    """
    Returns the count of each class as a dictionary
    to calculate accuracy
    """
    neighbors = getNeighbors(training_data, test_row, k)
    output_vals = [row[-1] for row in neighbors]

    counts = dict()

    for i in output_vals:
        counts[i] = counts.get(i, 0) + 1
    
    return counts
        
def main():

    script, pendigits_training, pendigits_test, k = argv

    #Append .txt to filename if it does not have an extension
    pendigits_training = (pendigits_training + '.txt') if len(pendigits_training.split('.')) == 1 else pendigits_training
    pendigits_test = (pendigits_test + '.txt') if len(pendigits_test.split('.')) == 1 else pendigits_test
    
    training_file = open(pendigits_training, 'r')
    test_file = open(pendigits_test, 'r')

    training_dataset = []
    for line in training_file:
        training_dataset.append([int(datapoint) for datapoint in line.split()])

    test_dataset = []
    for line in test_file:
        test_dataset.append([int(datapoint) for datapoint in line.split()])
    
    meanAndStd = getMeanAndStd(training_dataset)

    normalizeData(meanAndStd, training_dataset)
    normalizeData(meanAndStd, test_dataset)

    classification_accuracy = 0

    for i in range(len(test_dataset)):
        row = test_dataset[i]
        predicted_class = predictClass(training_dataset, row, k)
        true_class = row[-1]
        accuracy = 0

        counts = getCounts(training_dataset, row, k)
        v = [value for value in counts.values()]
    
        if(v.count(max(v)) == 1 and (predicted_class==true_class)):
            accuracy = 1
        elif(v.count(max(v)) > 1 and counts[predicted_class] == max(v)):
            accuracy = 1/v.count(max(v))
            
        print("ID={0:5d}, predicted={1:3d}, true={2:3d}, accuracy={3:5.2f}\n".format(i, predicted_class, true_class, accuracy))
        classification_accuracy += accuracy
        
    classification_accuracy = classification_accuracy/len(test_dataset)

    print("classification accuracy= {0:6.4f}\n".format(classification_accuracy))

    training_file.close()
    test_file.close()

if __name__ == "__main__":
    main()