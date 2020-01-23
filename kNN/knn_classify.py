from sys import argv
import math

script, pendigits_training, pendigits_test, k = argv

def getMeanAndStd(training_dataset):
    meanAndStd = []
    for i in range(len(training_dataset[0])-1):
        column = [row[i] for row in training_dataset]
        mean = sum(column)/len(column)
        sigma = 0
        for datapoint in column:
            sigma += abs((datapoint - mean))**2
        
        std = math.sqrt(sigma/len(column))
        meanAndStd.append({"mean": mean, "std": std})

    return meanAndStd

def normalizeData(training_dataset):
    meanAndStd = getMeanAndStd(training_dataset)
    for i in range(len(training_dataset)):
        for j in range(len(training_dataset[i])-1):
            mean = meanAndStd[j]["mean"]
            std = meanAndStd[j]["std"]
            training_dataset[i][j] = (training_dataset[i][j] - mean)/std

def euclidianDistance(row1, row2):
    dist = 0.0
    for i in range(len(row1)-1):
        dist += (row1[i] - row2[i])**2
    
    return math.sqrt(dist)

def getNeighbors(training_data, test_row):
    distances = list()
    for training_row in training_data:
        dist = euclidianDistance(training_row, test_row)
        distances.append([training_row, dist])

    distances.sort(key=lambda row:row[1])
    neighbors = list()

    for i in range(int(k)):
        neighbors.append(distances[i][0])

    return neighbors

def predictClass(training_data, test_row):
    neighbors = getNeighbors(training_data, test_row)
    output_vals = [row[-1] for row in neighbors]
    prediction = max(set(output_vals), key=output_vals.count)

    return prediction

def getCounts(training_data, test_row):
    neighbors = getNeighbors(training_data, test_row)
    output_vals = [row[-1] for row in neighbors]

    counts = dict()

    for i in output_vals:
        counts[i] = counts.get(i, 0) + 1
    
    return counts
        

def main():

    training_file = open(f"{pendigits_training}.txt", 'r')
    test_file = open(f"{pendigits_test}.txt", 'r')

    training_dataset = []
    for line in training_file:
        training_dataset.append([int(datapoint) for datapoint in line.split()])

    test_dataset = []
    for line in test_file:
        test_dataset.append([int(datapoint) for datapoint in line.split()])
    
    normalizeData(training_dataset)
    normalizeData(test_dataset)

    classification_accuracy = 0

    for i in range(len(test_dataset)):
        row = test_dataset[i]
        predicted_class = predictClass(training_dataset, row)
        true_class = row[-1]
        accuracy = 0

        counts = getCounts(training_dataset, row)
        v = [value for value in counts.values()]
    
        if(v.count(max(v)) == 1 and (predicted_class==true_class)):
            accuracy = 1
        elif(v.count(max(v)) > 1 and counts[predicted_class] == max(v)):
            accuracy = 1/v.count(max(v))

        classification_accuracy += accuracy
        print("ID={0:5d}, predicted={1:3d}, true={2:3d}, accuracy={3:5.2f}\n".format(i, predicted_class, true_class, accuracy))
    
    classification_accuracy = classification_accuracy/len(test_dataset)

    print("Classification Accuracy: {0:6.4f}".format(classification_accuracy))

if __name__ == "__main__":
    main()