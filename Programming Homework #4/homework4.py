import numpy as np
import csv
import random
import math
import operator
from decimal import *
from numpy import genfromtxt
import matplotlib.pyplot as plt

##### METHODS #####
# Performs z-scaling on matrix
# @param: matrix - The matrix/array to scale
# @return: matrix - The matrix that has been scaled by the mean of each
#                   column and the standard deviation of each column
def zscale(matrix, means,stdev):
    offset = len(means)-2
    matrix[:,0:offset] = (matrix[:,0:offset] - means[0:offset])/(stdev[0:offset])
    return matrix

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
            trainingSet.append(inputData1[i])
        else:
            testSet.append(inputData1[i])
    return trainingSet, testSet   
  
#method to initialize QR Decomposition on a matrix X
#@param: X - the matrix we want to decompose. We note that the
#           X is not an augmented matrix
def QRdecompose(X):
    
    # Copy
    R = X.copy()
    # Store the shape of A
    [m, n] = X.shape
    # Create identity matrix of size m
    Q = np.identity(m)
    
    # Applying method 1. Recursively define H
    for i in range(n - (m == n)):
        
        # Empty householder identity matrix
        H = np.identity(m)
        
        # Create householder matrix
        H[i:, i:] = householdervk(R[i:, i])
        
        # Recreate Q,R
        Q = np.dot(Q, H)
        R = np.dot(H, R)
    return [Q,R]
 
def householdervk(X):
    
    # Determine shift
    vk = X / (X[0] + np.copysign(np.linalg.norm(X), X[0]))
    
    # e1
    vk[0] = 1
    
    # Take the shape of each R[i:,i]
    H = np.identity(X.shape[0])
    
    # Create householder matrix
    H = H - (2 / np.dot(vk, vk)) * np.dot(vk[:, None], vk[None, :])
    return H
 
#method for back substitution a matrix A with b
#@param: R - Upper right triangular matrix already formatted
#@param: b - A nx1 vector
def backsolve(R,b):
    n = np.size(b)
    x = np.zeros((n,1))
    
    # Start at the end, solve accordingly
    for i in range(n-1,-1,-1):
        x[i] = (b[i] - np.dot(R[i,:],x))/R[i,i]
        
    return x

    
#method for adjusting to method 2. Remove (m-n) rows and adjust
#sizes accordingly
#@param: Q - a matrix
#@param: R - an upper right triangular matrix with rows of 0's
def simplifyQR(Q,R):
   [Rm,Rn] = R.shape
   # Remove rows of zeros and adjust Q matrix to match dimensions
   # for np.dot
   if Rm != Rn:
       R = R[0:Rn,:]
       Q = Q[:,0:Rn]
   return Q,R

#method for calculating the root mean squared error
#@param: YActual - the correct vector
#@param: YEstimated - the estimated vector after QR Decomposition
def rmse(Y_actual,Y_estimated):
    return np.sqrt(np.mean((Y_actual-Y_estimated)**2))


# Method to pick random centroids from data set
def randomCentroid(inputData, k):
    
    size = len(inputData)
    x1 = Decimal(k)/Decimal(size)

    centroids = [] #stores hit rates on index
    j = 0

    for i in range(0, size):
        # Compare with random uniform
        x2 = random.uniform(0,1)
        if x2 < x1:
            centroids.append(inputData[i])
            j = j+1
        # Update x1 to new conditional probability
        x1 = Decimal(k-j)/Decimal(size-i)
    return np.array(centroids)

# Param: X is 
def cluster(X, mu):
    clusters  = {}
    for x in X:
        mu2 = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
                    for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[mu2].append(x)
        except KeyError:
            clusters[mu2] = [x]
    return clusters
 
def updateMid(mu, clusters):
    mymu = []
    keys = sorted(clusters.keys())
    for k in keys:
        mymu.append(np.mean(clusters[k], axis = 0))
    return mymu
 
def converged(mu, old):
    return (set([tuple(a) for a in mu]) == set([tuple(a) for a in old]))
 
def find_centers(X, K):
    # Initialize to K random centers
    old = randomCentroid(X, K)
    mu = randomCentroid(X, K)
    while not converged(mu, old):
        old = mu
        # Assign all points in X to clusters
        clusters = cluster(X, mu)
        # Reevaluate centers
        mu = updateMid(old, clusters)
    return(mu, clusters)
def calculate_wss(mu,clusters):
    n = len(mu)
    total = 0
    for i in range(n):
        ithsum = 0
        for j in range(len(clusters[i])):
            ithsum = ithsum + sum((clusters[i][j] -mu[i])**2)
        total = total + ithsum
    return total
# Method to calculate the total rmse of a cluster
#@param: Y_actual - cluster
#@param: Y_estimated - cluster mean 
def calculate_total_rmse(rmse_array):
    total = 0
    for i in rmse_array:
        total = total + i
    return total
    
################## ABALONE DATA SET ##################
# Read CSV Files
inputFile = open('abalone.data')
inputReader = csv.reader(inputFile)
inputData = list(inputReader)   #inputData = list of our data (which is in lists)


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

for i in range(size):
    if inputData[i][0] == 'M':
        inputData[i][0] = 0
    elif inputData[i][0] == 'F':
        inputData[i][0] = 1
    elif inputData[i][0] == 'I':
        inputData[i][0] = 2

# Add 3 columns for classification of sex
proxiedData = np.zeros((size,len(inputData[0])+3))
proxiedData[:,:-3] = inputData
# Fix data for Euclidean distance
for i in range(size):
    if proxiedData[i][0] == 0:
        proxiedData[i][9] = 1
    elif proxiedData[i][0] == 1:
        proxiedData[i][10] = 1
    elif proxiedData[i][0] == 2:
        proxiedData[i][11] = 1

# Remove first column
proxiedData = proxiedData[:,1:12]

# Swap the actual to the predictions
proxiedData[:,[7,8,9,10]] = proxiedData[:,[8,9,10,7]]

# Keep track of counter and create training and testing sets
trainIndex = []
testIndex = []
trainingSet = []
testSet = []

# Create training and testing sets
for i in range(size):
    if counter[i] == 1:
        trainIndex.append(i)
        trainingSet.append(proxiedData[i])
    else:
        testIndex.append(i)
        testSet.append(proxiedData[i])
        

# Change to numeric arrays
trainingSet = np.array(trainingSet)
testSet = np.array(testSet)

X_train = trainingSet[:,:-1] 
Y_train= trainingSet[:,-1]
X_test = testSet[:,:-1]
Y_test = testSet[:,-1]

#ZSCALING
means = X_train.mean(axis=0)
stdevs = X_train.std(axis=0)
X_train_scaled = zscale(X_train, means, stdevs)
X_test_scaled = zscale(X_test,means,stdevs)

train_means = trainingSet.mean(axis=0)
train_stdevs = trainingSet.std(axis=0)

train_set_scaled = zscale(trainingSet,train_means,train_stdevs)
test_set_scaled = zscale(testSet,train_means,train_stdevs)

fmean = []
fstd = []
fmean1 = []
fstd1 = []

wcssArray = []
rmseArray = []

# K = 1
test1,test2 = find_centers(train_set_scaled,1)
error1 = calculate_wss(test1,test2)

# Fix dimension
Y_test = Y_test[:,np.newaxis]

print ("K = 1")
print ("centroids = ", test1)
print ("WCSS = ", error1)
for k in range(0, len(test2)):
    for j in range(0,len(test2[k][0])):
        a = np.array(test2[k])
        b = a[:,j]
        a2 = np.array(b)
        fmean.append(np.mean(a2))
        fstd.append(np.std(a2))
        fmean1.append(fmean[0])
        fstd1.append(fstd[0])
        fmean = []
        fstd = []
    print ("cluster #",k+1)
    print ("MEAN =", fmean1)
    fmean1 = []
    print ("STDEV =", fstd1)
    fstd1 = []
    
totalRMSE = 0
for k in range(0, len(test2)):
    a = np.array(test2[k])
    
    # Perform QR Decomposition and backsolving
    Q,R = QRdecompose(a[:,:-1])
    Y_train = a[:,-1]
    # Check if QR decomposition was successful
    np.dot(Q,R)

    # Fixing and backsolving
    Q,R = simplifyQR(Q,R)
    Z = np.dot(Q.T,Y_train)
    beta = backsolve(R,Z)

    RMSE = rmse(np.dot(X_test,beta), Y_test)
    print ("RMSE IS ", RMSE)
    totalRMSE = totalRMSE + RMSE

print ("RMSE = ", totalRMSE)
wcssArray.append(error1)
rmseArray.append(totalRMSE)
print ("-------------------------------------------------------------")

# K = 2
test1,test2 = find_centers(train_set_scaled,2)
error1 = calculate_wss(test1,test2)
print ("K = 2")
print ("centroids = ", test1)
print ("WCSS = ", error1)
for k in range(0, len(test2)):
    for j in range(0,len(test2[k][0])):
        a = np.array(test2[k])
        b = a[:,j]
        a2 = np.array(b)
        fmean.append(np.mean(a2))
        fstd.append(np.std(a2))
        fmean1.append(fmean[0])
        fstd1.append(fstd[0])
        fmean = []
        fstd = []
    print ("cluster #",k+1)
    print ("MEAN =", fmean1)
    fmean1 = []
    print ("STDEV =", fstd1)
    fstd1 = []

totalRMSE = 0
for k in range(0, len(test2)):
    a = np.array(test2[k])
    
    # Perform QR Decomposition and backsolving
    Q,R = QRdecompose(a[:,:-1])
    Y_train = a[:,-1]
    # Check if QR decomposition was successful
    np.dot(Q,R)

    # Fixing and backsolving
    Q,R = simplifyQR(Q,R)
    Z = np.dot(Q.T,Y_train)
    beta = backsolve(R,Z)

    RMSE = rmse(np.dot(X_test,beta), Y_test)
    print ("RMSE IS ", RMSE)
    totalRMSE = totalRMSE + RMSE

print ("RMSE = ", totalRMSE)
wcssArray.append(error1)
rmseArray.append(totalRMSE)
print ("-------------------------------------------------------------")

# K = 4
test1,test2 = find_centers(train_set_scaled,4)
error1 = calculate_wss(test1,test2)
print ("K = 4")
print ("centroids = ", test1)
print ("WCSS = ", error1)
for k in range(0, len(test2)):
    for j in range(0,len(test2[k][0])):
        a = np.array(test2[k])
        b = a[:,j]
        a2 = np.array(b)
        fmean.append(np.mean(a2))
        fstd.append(np.std(a2))
        fmean1.append(fmean[0])
        fstd1.append(fstd[0])
        fmean = []
        fstd = []
    print ("cluster #",k+1)
    print ("MEAN =", fmean1)
    fmean1 = []
    print ("STDEV =", fstd1)
    fstd1 = []

totalRMSE = 0
for k in range(0, len(test2)):
    a = np.array(test2[k])
    
    # Perform QR Decomposition and backsolving
    Q,R = QRdecompose(a[:,:-1])
    Y_train = a[:,-1]
    # Check if QR decomposition was successful
    np.dot(Q,R)

    # Fixing and backsolving
    Q,R = simplifyQR(Q,R)
    Z = np.dot(Q.T,Y_train)
    beta = backsolve(R,Z)

    RMSE = rmse(np.dot(X_test,beta), Y_test)
    print ("RMSE IS ", RMSE)
    totalRMSE = totalRMSE + RMSE

print ("RMSE = ", totalRMSE)
wcssArray.append(error1)
rmseArray.append(totalRMSE)
print ("-------------------------------------------------------------")

# K = 8
test1,test2 = find_centers(train_set_scaled,8)
error1 = calculate_wss(test1,test2)
print ("K = 8")
print ("centroids = ", test1)
print ("WCSS = ", error1)
for k in range(0, len(test2)):
    for j in range(0,len(test2[k][0])):
        a = np.array(test2[k])
        b = a[:,j]
        a2 = np.array(b)
        fmean.append(np.mean(a2))
        fstd.append(np.std(a2))
        fmean1.append(fmean[0])
        fstd1.append(fstd[0])
        fmean = []
        fstd = []
    print ("cluster #",k+1)
    print ("MEAN =", fmean1)
    fmean1 = []
    print ("STDEV =", fstd1)
    fstd1 = []

totalRMSE = 0
for k in range(0, len(test2)):
    a = np.array(test2[k])
    
    # Perform QR Decomposition and backsolving
    Q,R = QRdecompose(a[:,:-1])
    Y_train = a[:,-1]
    # Check if QR decomposition was successful
    np.dot(Q,R)

    # Fixing and backsolving
    Q,R = simplifyQR(Q,R)
    Z = np.dot(Q.T,Y_train)
    beta = backsolve(R,Z)

    RMSE = rmse(np.dot(X_test,beta), Y_test)
    print ("RMSE IS ", RMSE)
    totalRMSE = totalRMSE + RMSE

print ("RMSE = ", totalRMSE)
wcssArray.append(error1)
rmseArray.append(totalRMSE)
print ("-------------------------------------------------------------")

# K = 16
test1,test2 = find_centers(train_set_scaled,16)
error1 = calculate_wss(test1,test2)
print ("K = 16")
print ("centroids = ", test1)
print ("WCSS = ", error1)
for k in range(0, len(test2)):
    for j in range(0,len(test2[k][0])):
        a = np.array(test2[k])
        b = a[:,j]
        a2 = np.array(b)
        fmean.append(np.mean(a2))
        fstd.append(np.std(a2))
        fmean1.append(fmean[0])
        fstd1.append(fstd[0])
        fmean = []
        fstd = []
    print ("cluster #",k+1)
    print ("MEAN =", fmean1)
    fmean1 = []
    print ("STDEV =", fstd1)
    fstd1 = []

totalRMSE = 0
for k in range(0, len(test2)):
    a = np.array(test2[k])
    
    # Perform QR Decomposition and backsolving
    Q,R = QRdecompose(a[:,:-1])
    Y_train = a[:,-1]
    # Check if QR decomposition was successful
    np.dot(Q,R)

    # Fixing and backsolving
    Q,R = simplifyQR(Q,R)
    Z = np.dot(Q.T,Y_train)
    beta = backsolve(R,Z)

    RMSE = rmse(np.dot(X_test,beta), Y_test)
    print ("RMSE IS ", RMSE)
    totalRMSE = totalRMSE + RMSE

print ("RMSE = ", totalRMSE)
wcssArray.append(error1)
rmseArray.append(totalRMSE)
print ("-------------------------------------------------------------")

print ("WCSS array = ", wcssArray)
print ("RMSE array = ", rmseArray)

plt.figure(0)
plt.plot([1,2,4,8,16], wcssArray,'ro')
plt.title('K vs WCSS')
plt.xlabel('K')
plt.ylabel('WCSS')
plt.show()

plt.figure(1)
plt.plot([1,2,4,8,16], rmseArray,'ro')
plt.title('K vs RMSE')
plt.xlabel('K')
plt.ylabel('RMSE')
plt.show()
# End
