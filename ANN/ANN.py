import os


dataset = []
filename = os.getcwd() + r'/ANN/data/optdigits-3.tra'


def create_dataset(filename):
    f = open(filename, 'r')
    for line in f:
        line = line.rstrip('\n')
        dataset.append([line])
    print(dataset)

create_dataset(filename)
