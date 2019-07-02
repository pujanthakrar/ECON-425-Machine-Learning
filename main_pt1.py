#!/usr/bin/env python
# coding: utf-8

# In[190]:


import numpy as np
import matplotlib.pyplot as plt
from getDataset import getDataSet
from sklearn.linear_model import LogisticRegression

import math
import pandas as pd

from pandas import DataFrame
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from numpy import loadtxt, where
from pylab import scatter, show, legend, xlabel, ylabel

from util1 import Cost_Function, Gradient_Descent, Cost_Function_Derivative, Cost_Function, Prediction, Sigmoid



# Starting codes

# Fill in the codes between "%PLACEHOLDER#start" and "PLACEHOLDER#end"

# step 1: generate dataset that includes both positive and negative samples,
# where each sample is described with two features.
# 250 samples in total.

[X, y] = getDataSet()  # note that y contains only 1s and 0s,

# create figure for all charts to be placed on so can be viewed together
fig = plt.figure()


def func_DisplayData(dataSamplesX, dataSamplesY, chartNum, titleMessage):
    idx1 = (dataSamplesY == 0).nonzero()  # object indices for the 1st class
    idx2 = (dataSamplesY == 1).nonzero()
    ax = fig.add_subplot(1, 3, chartNum)
    # no more variables are needed
    plt.plot(dataSamplesX[idx1, 0], dataSamplesX[idx1, 1], 'r*')
    plt.plot(dataSamplesX[idx2, 0], dataSamplesX[idx2, 1], 'b*')
    # axis tight
    ax.set_xlabel('x_1')
    ax.set_ylabel('x_2')
    ax.set_title(titleMessage)


# plotting all samples
func_DisplayData(X, y, 1, 'All samples')

# number of training samples
nTrain = 120

######################PLACEHOLDER 1#start#########################
# write you own code to randomly pick up nTrain number of samples for training and use the rest for testing.
# WARNIN: 

maxIndex = len(X)
randomTrainingSamples = np.random.choice(maxIndex, nTrain, replace=False)

trainX =  X[randomTrainingSamples,:]#  training samples
trainY =  y[randomTrainingSamples,:]# labels of training samples    nTrain X 1

#find indices for data not in the trainins samples
indexes = np.arange(0,250,1)
new = [x for x in indexes if x not in randomTrainingSamples]

testX =   X[new,:]# testing samples               
testY =   y[new,:] # labels of testing samples     nTest X 1

####################PLACEHOLDER 1#end#########################

# plot the samples you have pickup for training, check to confirm that both negative
# and positive samples are included.
func_DisplayData(trainX, trainY, 2, 'training samples')
func_DisplayData(testX, testY, 3, 'testing samples')

# show all charts
plt.show()


#  step 2: train logistic regression models


######################PLACEHOLDER2 #start#########################
# in this placefolder you will need to train a logistic model using the training data: trainX, and trainY.

clf = LogisticRegression()
# call the function fit() to train the class instance
clf.fit(trainX,trainY)


# visualize data using functions in the library pylab 
pos = where(y == 1)
neg = where(y == 0)
scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
xlabel('Feature 1: score 1')
ylabel('Feature 2: score 2')
legend(['Label:  Admitted', 'Label: Not Admitted'])
show()

##Gradient descent method to learn logistic regression
theta = [0,0] #initial model parameters
alpha = 0.1 # learning rates
max_iteration = 1000 # maximal iterations


m = len(y) # number of samples

for x in range(max_iteration):
        # call the functions for gradient descent method
        new_theta = Gradient_Descent(X,y,theta,m,alpha)
        theta = new_theta
        if x % 200 == 0:
            # calculate the cost function with the present theta
            Cost_Function(X,y,theta,m)
            print('theta ', theta)
            print('cost is ', Cost_Function(X,y,theta,m))
            

#get score for sklearn method
score = 0
winner = ""
# accuracy for sklearn
scikit_score = clf.score(testX,testY)
length = len(testX)
for i in range(length):
        prediction = round(Prediction(testX[i],theta))
        answer = testY[i]
        if prediction == answer:
            score += 1

#get score for gradient descent method and compare to sk learn
my_score = float(score) / float(length)
if my_score > scikit_score:
        print('You won!')
elif my_score == scikit_score:
        print('Its a tie!')
else:
        print('Scikit won.. :(')
print('Your score: ', my_score)
print('Scikits score: ', scikit_score) 

######################PLACEHOLDER2 #end #########################

 
 
# step 3: Use the model to get class labels of testing samples.

######################PLACEHOLDER3 #start#########################
# codes for making prediction, 
# with the learned model, apply the logistic model over testing samples
# hatProb is the probability of belonging to the class 1.
# y = 1/(1+exp(-Xb))
# yHat = 1./(1+exp( -[ones( size(X,1),1 ), X] * bHat )); ));
# WARNING: please DELETE THE FOLLOWING CODEING LINES and write your own codes for making predictions

#prediction with gradient descent method

predGD = [Prediction(i,theta) for i in testX]
predGD = [float(int(j>=.6)) for j in predGD]

#prediction with sklearn

predSK = clf.predict(testX)

print("GD method: ", predGD)
print("sklearn method: ", predSK)
######################PLACEHOLDER 3 #end #########################


# step 4: evaluation
# compare predictions and true labels to calculate average error and standard deviation
testYDiff = np.abs(predGD - testY)
avgErr = np.mean(testYDiff)
stdErr = np.std(testYDiff)

testYDiff1 = np.abs(predSK - testY)
avgErr1 = np.mean(testYDiff1)
stdErr1 = np.std(testYDiff1)

print('GD average error: {} ({})'.format(avgErr, stdErr))

print('SK average error: {} ({})'.format(avgErr1, stdErr1))


#Problem 3 confusion matrix

#function should return accuracy, per class precision, per class recall rate
#for a 2 by two matrix with ground truth labels 0 and 1

 
def func_calConfusionMatrix(predY,trueY):
    uniqueVals = [0,1]
    matrix=np.zeros((2,3))
    for j in uniqueVals :
        #for value 0
        trueVal = 0
        falseVal = 0
        
        for i in range(len(predY)):
            if(trueY[i][0]==j):
                if(int(predY[i])==int(j)):
                    trueVal = trueVal+1
                else:
                    falseVal = falseVal+1
        if(j==0):
            matrix[j]=[j,trueVal,falseVal]
        else:
            matrix[j]=[j,falseVal,trueVal]
        
    accuracy = (matrix[0,1]+matrix[1,2])/len(predY)
    prec_0 = matrix[0,1]/(matrix[0,1]+matrix[1,1])
    prec_1 = matrix[1,2]/(matrix[1,2]+matrix[0,2])
    recall_0 = matrix[0,1]/(matrix[0,1]+matrix[0,2])
    recall_1 = matrix[1,2]/(matrix[1,1]+matrix[1,2])
    print("Confusion Matrix:")
    print(matrix)
    print('Accuracy:',accuracy)
    print('Precision 0:', prec_0)
    print('Precision 1:', prec_1)
    print('Recall 0:', recall_0)
    print('Recall 1:', recall_1)
 

print("SK Learn Confusion Matrix:")
func_calConfusionMatrix(predSK,testY)

        
print("GD Confusion Matrix:")
func_calConfusionMatrix(predGD,testY)

