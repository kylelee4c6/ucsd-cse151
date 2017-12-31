import numpy as np
import csv
import random
import math
import operator
from decimal import *
from numpy import genfromtxt

##### METHODS #####
# Performs z-scaling on matrix
# @param: matrix - The matrix/array to scale
# @return: matrix - The matrix that has been scaled by the mean of each
#                   column and the standard deviation of each column
def zscale(matrix, means,stdev):
    offset = len(means)
    matrix[:,0:offset] = (matrix[:,0:offset] - means[0:offset])/(stdev[0:offset])
    return matrix

# Create a confusion matrix
# @param: predicted - Predictions in array
# @param: actual - Actual values
# @param: size - Number of classes
def confuseMe(predicted, actual,size):
    # Initialize confusion matrix
    confusionMatrix = [[0 for x in range(size)] for x in range(size)]
    confusionMatrix = np.array(confusionMatrix).astype(np.float)
    # Count true positives and false positives
    for i in range(len(predicted)):
        # If actual is equal to predicted, increase the diagonal by 1
        if actual[i] == predicted[i]:
            confusionMatrix[int(actual[i])][int(actual[i])] = confusionMatrix[int(actual[i])][int(actual[i])] + 1

        # If actual does not equal predicted, increase that respective spot by 1
        elif actual[i] is not predicted[i]:
            confusionMatrix[int(actual[i])][int(predicted[i])] = confusionMatrix[int(actual[i])][int(predicted[i])] + 1

    return confusionMatrix

# Change a confusion matrix into a series of ratios
# @param: confusionMatrix - The Confusion Matrix to change
def decimateConfusionMatrix(confusionMatrix):
    for k in range(len(confusionMatrix)):
        weight = np.sum(confusionMatrix[k,:])
        if (weight > 0):
            confusionMatrix[k,:] = confusionMatrix[k,:]/weight
    return confusionMatrix


# Calculate the distance between two instances of data
# @param: data1 - First data to compare to
# @param: data2 - Second data to compare to
# @param: length - Size of data
def calculateDistance(data1, data2, length):
    distance = 0
    for x in range(length):
        distance+=pow((data1[x] - data2[x]),2)
    return math.sqrt(distance)
#method for generating hits
#@param:x => our data
#      :testSize => hit rate
#      :genList => list of # of hits per index
#@return:genList => updated list with hits per index
def count(x, testSize, genList):
    size = len(x)
    expectedDraws = int(round(size*testSize))
    x1 = Decimal(expectedDraws)/Decimal(size)

    j = 0

    for i in range(0, size):
        # Compare with random uniform
        x2 = random.uniform(0,1)
        if x2 < x1:
            genList[i] = genList[i] + 1
            j = j+1
        # Update x1 to new conditional probability
        x1 = Decimal(expectedDraws-j)/Decimal(size-i)
    return genList


#method for separating input data based on counter
#@param: counter => a vector that stores the indices of test/train sets
#@param: inputData => original data
#@return: trainingSet - Training Set based on data
#@return: testSet - Test Set based on data
def separateSet(counter,inputData):
    trainingSet = []
    testSet = []
    size = len(inputData)
    for i in range(size):
        if counter[i] == 1:
            trainingSet.append(inputData[i])
        else:
            testSet.append(inputData[i])
    return trainingSet, testSet


def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated


def summarize(dataset):
	summaries = [(sum(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries

def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.items():
		summaries[classValue] = summarize(instances)
	return summaries

# Assume Normal(mean,stdev)
def calculateProbability(x, totals):
    return (float(x+1))/float(totals+1)


def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            x = inputVector[i]
            totals = classSummaries[i]
            probabilities[classValue] *= calculateProbability(x,totals)
    return probabilities

def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.items():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions

def getAccuracy(testSet, predictions):
	correct = 0
	for i in range(len(testSet)):
		if testSet[i][-1] == predictions[i]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0
# Calculate probability

################## SPAM PRUNED DATA SET ##################
# Read CSV Files
inputData = genfromtxt(r'C:\Users\Kyle Lee\Google Drive\School Doc. 2015-2016\CSE 151\Programming Homework #5\SpamDataPruned.csv', delimiter =',')


# Initiaize test Size
trainingSize = 0.9

# Length of the data
size = len(inputData)

# Initialize counter
counter = [] #stores hit rates on index
# initialize counter array
for x in range(size):
    counter.append(0)

# Find test size
for i in range(1,2):
    counter = count(inputData, trainingSize, counter)

# Create training and testing sets
[trainingSet, testSet] = separateSet(counter, inputData)
# Change to numeric arrays
trainingSet = np.array(trainingSet)
testSet = np.array(testSet)

summaries = summarizeByClass(trainingSet)

X_train = trainingSet[:,:-1]
Y_train= trainingSet[:,-1]

X_test = testSet[:,:-1]
Y_test = testSet[:,-1]
#test model
predictions = getPredictions(summaries, testSet)
accuracy = getAccuracy(testSet,predictions)
print('Accuracy:').format(accuracy)
