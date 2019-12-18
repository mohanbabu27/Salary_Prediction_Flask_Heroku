# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 22:48:04 2019

@author: tallurim
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('C:/Users/tallurim/Desktop/ML/Deployment_Flask_Master/hiring.csv')

dataset['experience'].fillna(0, inplace=True) #FE replace all nan values as 0

dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True) #filling data sent with mean value

X = dataset.iloc[:, :3] #We have 3 features in data set which are independant values

def convert_to_int(word):  #Converting words to inegers in exp feature
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6,
                 'seven':7, 'eight':8, 'nine':9, 'ten':10, 'eleven':11,
                 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]

#apply word to int in to feature column using lambda function
X['experience'] = X['experience'].apply(lambda x:  convert_to_int(x))

#Assigning the Y value
y = dataset.iloc[:, -1]

#Splitting training and Test Set
#Since its very small data set, will train our model with all available data

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with training data
regressor.fit(X, y)

#Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

#loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2,9,6]]))



